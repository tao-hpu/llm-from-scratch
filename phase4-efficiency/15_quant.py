"""
里程碑 15(第 17 章):量化 —— 把第 6 章训出来的 124M 从 32 位压到 4 位,看它掉多少
==============================================================================

模型权重默认是 32 位浮点。量化 = 用更少的位数存每个权重:

  【步骤 1】scale   找一个缩放系数,把一段权重的范围映射到整数格点上(比如 4 位只有 16 个格)
  【步骤 2】round   每个权重四舍五入到最近的格点,存格点编号
  【步骤 3】dequant 用的时候乘回 scale,得到近似值

全部精度损失都出在"一个 scale 管多少个权重":
  per-tensor   整个矩阵共用一个 scale → 一个离群值就把所有格子撑大
  per-channel  每一行一个 scale
  group-g      每 g 个权重一个 scale(llama.cpp 的 Q4_K、NVIDIA 的 NVFP4 都是这一类)
scale 本身也要占位数,所以"每个权重平均几位"= 格点位数 + scale 摊下来的位数。

这里做的是"假量化":量化后立刻反量化回浮点再算 loss。数值结果和真 4 位 kernel 一致,
只是不会真的省显存、变快 —— 那需要专门的算子,不改变这里比较的东西。

跑法:
    python 15_quant.py                     # 需要 ../phase1-124m/ckpt10b/latest.pt 和一份 FineWeb-Edu val shard
    python 15_quant.py --tokens 65536      # 少评一点,跑得快
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=str, default=os.path.join(HERE, "..", "phase1-124m", "ckpt10b", "latest.pt"))
p.add_argument("--data", type=str, default=os.path.join(HERE, "data", "edufineweb_val_000000.npy"))
p.add_argument("--tokens", type=int, default=131072, help="评测用多少个 val token(按 1024 一段切)")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--json", type=str, default="")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ---- 模型定义(与 05_sample.py 一致,去掉 KV cache)----
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln_2, self.mlp = nn.LayerNorm(cfg.n_embd), MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.shape
        x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(T, device=idx.device))
        for blk in self.transformer.h:
            x = blk(x)
        return self.lm_head(self.transformer.ln_f(x))

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
cfg = GPTConfig(**ckpt["config"])
model = GPT(cfg)
model.load_state_dict(ckpt["model"])
model.eval().to(device)
ORIG = {k: v.detach().clone() for k, v in model.state_dict().items()}
print(f"device={device} | 加载 {os.path.relpath(args.ckpt, HERE)}(step {ckpt.get('step')}, 训练时 val {ckpt.get('val_loss')})")

# 要量化的矩阵:12 层里所有 Linear 的权重(c_attn / c_proj / c_fc / mlp.c_proj)。
# 词嵌入(和输出头共用)、位置嵌入、LayerNorm、bias 保持 16 位 —— 常见做法,单独再测一组"连嵌入一起量化"。
QKEYS = [k for k, v in ORIG.items() if k.startswith("transformer.h.") and k.endswith(".weight") and v.dim() == 2]
EMB_KEY = "transformer.wte.weight"

# ---- 评测数据:FineWeb-Edu val shard 开头的若干段 1024 token ----
tokens = np.load(args.data, mmap_mode="r")
n_seq = args.tokens // cfg.block_size
buf = torch.from_numpy(np.array(tokens[: n_seq * cfg.block_size + 1], dtype=np.int64))
X = buf[:-1].view(n_seq, cfg.block_size)
Y = buf[1:].view(n_seq, cfg.block_size)

@torch.no_grad()
def val_loss():
    tot = 0.0
    for i in range(0, n_seq, args.batch):
        logits = model(X[i:i + args.batch].to(device))
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), Y[i:i + args.batch].to(device).view(-1),
                               reduction="sum").item()
    return tot / (n_seq * cfg.block_size)

# ===========================================================================
# 量化格式。每个函数:输入原始权重 w,返回 (反量化后的 w, 每个权重平均占几位)
# ===========================================================================
def int_absmax(w, bits, group=None):
    """对称整数量化。group=None:整个矩阵一个 scale;'row':每行一个;整数 g:每 g 个一个。"""
    qmax = 2 ** (bits - 1) - 1
    rows, cols = w.shape
    if group is None:
        s = w.abs().max() / qmax
        return (w / s).round().clamp(-qmax, qmax) * s, bits + 16 / w.numel()
    if group == "row":
        s = w.abs().amax(dim=1, keepdim=True) / qmax
        return (w / s).round().clamp(-qmax, qmax) * s, bits + 16 / cols
    pad = (-cols) % group
    wg = F.pad(w, (0, pad)).view(rows, -1, group)
    s = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    deq = ((wg / s).round().clamp(-qmax, qmax) * s).view(rows, -1)[:, :cols]
    return deq, bits + 16 / group

def q4_k_like(w):
    """仿 llama.cpp Q4_K:每 256 个权重一个超块,切成 8 个 32 的子块。
    子块存 4 位非对称格点(0..15)+ 自己的 scale 和 min(各 6 位);超块存两个 16 位浮点把 6 位 scale/min 还原。
    每个权重 4 + (6+6)/32 + (16+16)/256 = 4.5 位。"""
    rows, cols = w.shape
    flat = w.reshape(-1)
    pad = (-flat.numel()) % 256
    x = F.pad(flat, (0, pad)).view(-1, 8, 32)                         # (超块, 子块, 32)
    lo = x.amin(-1).clamp(max=0)                                      # 子块最小值(≤0)
    hi = x.amax(-1)
    sc = (hi - lo) / 15                                               # 子块 scale
    mn = -lo                                                          # 子块 min(存正数)
    d = sc.amax(-1, keepdim=True).clamp(min=1e-12) / 63               # 超块:把 6 位 scale 还原的系数
    dmin = mn.amax(-1, keepdim=True).clamp(min=1e-12) / 63
    sc_q = (sc / d).round().clamp(0, 63)
    mn_q = (mn / dmin).round().clamp(0, 63)
    s = (sc_q * d).unsqueeze(-1).clamp(min=1e-12)
    m = (mn_q * dmin).unsqueeze(-1)
    q = ((x + m) / s).round().clamp(0, 15)
    deq = (q * s - m).view(-1)[: flat.numel()].view(rows, cols)
    return deq, 4.5

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])          # FP4 能表示的非负值

def round_to_grid(x, grid):
    """把 |x| 四舍五入到 grid 里最近的值,符号保留。"""
    g = grid.to(x.device)
    idx = (x.abs().unsqueeze(-1) - g).abs().argmin(-1)
    return g[idx] * x.sign()

def e4m3_grid():
    vals = [m / 8 * 2 ** -6 for m in range(8)]                          # 指数位全 0:次正规数
    for e in range(1, 16):
        for m in range(8):
            if e == 15 and m == 7:
                continue                                                # 这个编码留给 NaN
            vals.append((1 + m / 8) * 2 ** (e - 7))
    return torch.tensor(sorted(set(vals)))
E4M3 = e4m3_grid()                                                      # 最大 448

def nvfp4(w):
    """NVFP4:每 16 个值一个 FP8(E4M3)scale,外加整个张量一个 FP32 scale。元素是 E2M1。
    每个权重 4 + 8/16 = 4.5 位。"""
    rows, cols = w.shape
    flat = w.reshape(-1)
    pad = (-flat.numel()) % 16
    x = F.pad(flat, (0, pad)).view(-1, 16)
    t_scale = x.abs().max() / (6 * 448)                                 # 让块 scale 落进 E4M3 的范围
    b_scale = round_to_grid(x.abs().amax(-1, keepdim=True) / 6 / t_scale, E4M3) * t_scale
    b_scale = b_scale.clamp(min=1e-12)
    deq = round_to_grid((x / b_scale).clamp(-6, 6), E2M1) * b_scale
    return deq.view(-1)[: flat.numel()].view(rows, cols), 4 + 8 / 16

def mxfp4(w):
    """MXFP4(OCP Microscaling):每 32 个值一个 E8M0 scale(只能是 2 的整数次幂),元素是 E2M1。
    共享指数 = floor(log2(块内最大绝对值)) − 2,超出 ±6 的直接截断。每个权重 4 + 8/32 = 4.25 位。"""
    rows, cols = w.shape
    flat = w.reshape(-1)
    pad = (-flat.numel()) % 32
    x = F.pad(flat, (0, pad)).view(-1, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    s = 2.0 ** (torch.floor(torch.log2(amax)) - 2)
    deq = round_to_grid((x / s).clamp(-6, 6), E2M1) * s
    return deq.view(-1)[: flat.numel()].view(rows, cols), 4 + 8 / 32

def int4_keep_outliers(w, frac=0.001):
    """整个矩阵一个 scale 的 4 位,但把绝对值最大的 0.1% 权重原样留在 16 位(另存位置 32 位)。"""
    k = max(1, int(frac * w.numel()))
    thr = w.abs().reshape(-1).kthvalue(w.numel() - k).values
    out_mask = w.abs() > thr
    inner = torch.where(out_mask, torch.zeros_like(w), w)
    deq, _ = int_absmax(inner, 4)
    deq = torch.where(out_mask, w, deq)
    return deq, 4 + 16 / w.numel() + frac * (16 + 32)

FORMATS = [
    ("fp32",               "32 位浮点(原始)",                  None),
    ("fp16",               "16 位浮点",                          lambda w: (w.half().float(), 16)),
    ("int8_tensor",        "int8 · 整个矩阵一个 scale",          lambda w: int_absmax(w, 8)),
    ("int8_row",           "int8 · 每行一个 scale",              lambda w: int_absmax(w, 8, "row")),
    ("int4_tensor",        "int4 · 整个矩阵一个 scale",          lambda w: int_absmax(w, 4)),
    ("int4_tensor_outlier","int4 · 整矩阵 scale + 留 0.1% 离群值", int4_keep_outliers),
    ("int4_row",           "int4 · 每行一个 scale",              lambda w: int_absmax(w, 4, "row")),
    ("int4_g128",          "int4 · 每 128 个一个 scale",         lambda w: int_absmax(w, 4, 128)),
    ("int4_g32",           "int4 · 每 32 个一个 scale",          lambda w: int_absmax(w, 4, 32)),
    ("q4_k",               "Q4_K 式 · 超块 256 / 子块 32 / 非对称", q4_k_like),
    ("nvfp4",              "NVFP4 · E2M1 + 每 16 个一个 E4M3 scale", nvfp4),
    ("mxfp4",              "MXFP4 · E2M1 + 每 32 个一个 2 的幂 scale", mxfp4),
    ("int3_g32",           "int3 · 每 32 个一个 scale",          lambda w: int_absmax(w, 3, 32)),
    ("int2_g32",           "int2 · 每 32 个一个 scale",          lambda w: int_absmax(w, 2, 32)),
]

def apply(fn, keys):
    """把 keys 里的矩阵换成量化版,返回这些矩阵平均每个权重的位数。"""
    sd = {k: v.clone() for k, v in ORIG.items()}
    bits_sum = n_sum = 0
    for k in keys:
        w = ORIG[k].float()
        deq, bpw = fn(w)
        sd[k] = deq.to(ORIG[k].dtype)
        if k == EMB_KEY:
            sd["lm_head.weight"] = sd[k]      # 输出头和词嵌入是同一个张量,两个键都要换,否则后加载的会把量化结果盖回去
        bits_sum += bpw * w.numel()
        n_sum += w.numel()
    model.load_state_dict(sd)
    return bits_sum / n_sum

# 模型总大小:量化的矩阵按 bpw 算,其余参数(嵌入 / LayerNorm / bias)按 16 位算
N_Q = sum(ORIG[k].numel() for k in QKEYS)
N_ALL = sum(v.numel() for k, v in ORIG.items() if k != "lm_head.weight")     # 输出头与词嵌入共用,不重复算
N_EMB = ORIG[EMB_KEY].numel()

enc = tiktoken.get_encoding("gpt2")
PROMPT = enc.encode("The history of ancient Rome")

@torch.no_grad()
def greedy(n=32):
    seq = list(PROMPT)
    for _ in range(n):
        logits = model(torch.tensor([seq[-cfg.block_size:]], device=device))
        seq.append(int(logits[0, -1].argmax()))
    return enc.decode(seq[len(PROMPT):])

rows = []
base = None
t0 = time.time()
for name, label, fn in FORMATS:
    if fn is None:
        model.load_state_dict(ORIG)
        bpw = 32.0
    else:
        bpw = apply(fn, QKEYS)
    other_bits = 32 if name == "fp32" else 16
    mb = (N_Q * bpw + (N_ALL - N_Q) * other_bits) / 8 / 1e6
    loss = val_loss()
    base = base or loss
    row = {"name": name, "label": label, "bits_per_weight": round(bpw, 3), "model_mb": round(mb, 1),
           "val_loss": round(loss, 4), "delta": round(loss - base, 4), "sample": greedy()}
    rows.append(row)
    print(f"{label:<34s} {bpw:6.3f} 位 | {mb:6.1f} MB | val {loss:.4f}(+{loss-base:.4f})"
          f" | {row['sample'][:60]!r} | {time.time()-t0:.0f}s")

# ---- 额外一组:词嵌入 / 输出头也用 int4 g32 ----
bpw = apply(lambda w: int_absmax(w, 4, 32), QKEYS + [EMB_KEY])
mb = ((N_Q + N_EMB) * bpw + (N_ALL - N_Q - N_EMB) * 16) / 8 / 1e6
loss = val_loss()
rows.append({"name": "int4_g32_with_emb", "label": "int4 · 每 32 个一个 scale · 连词嵌入一起", "bits_per_weight": round(bpw, 3),
             "model_mb": round(mb, 1), "val_loss": round(loss, 4), "delta": round(loss - base, 4), "sample": greedy()})
print(f"int4 g32 连词嵌入: {mb:.1f} MB | val {loss:.4f}")

# ---- 给页面用:一个矩阵的权重分布、离群值、同一行在几种格式下的近似值 ----
W = ORIG["transformer.h.6.mlp.c_fc.weight"].float().cpu()
std = W.std().item()
hist_range = 8 * std
hist = torch.histc(W.clamp(-hist_range, hist_range), bins=80, min=-hist_range, max=hist_range)
outlier = {"std": round(std, 5), "absmax": round(W.abs().max().item(), 5),
           "absmax_over_std": round(W.abs().max().item() / std, 1),
           "frac_over_6std": round((W.abs() > 6 * std).float().mean().item(), 6),
           "frac_over_3std": round((W.abs() > 3 * std).float().mean().item(), 5)}
row_w = W[123:124, :32]
examples = {"orig": [round(v, 5) for v in row_w[0].tolist()]}
full_row = W[123:124]
for name, fn in [("int4_tensor", lambda w: int_absmax(w, 4)), ("int4_row", lambda w: int_absmax(w, 4, "row")),
                 ("int4_g32", lambda w: int_absmax(w, 4, 32)), ("nvfp4", nvfp4)]:
    if name == "int4_tensor":
        s = W.abs().max() / 7
        deq = ((row_w / s).round().clamp(-7, 7) * s)
    else:
        deq = fn(full_row)[0][:, :32]
    examples[name] = [round(v, 5) for v in deq[0].tolist()]
per_layer = []
for k in QKEYS:
    w = ORIG[k].float()
    per_layer.append({"key": k.replace("transformer.", ""), "absmax_over_std": round((w.abs().max() / w.std()).item(), 1)})

result = {"ckpt_step": ckpt.get("step"), "eval_tokens": n_seq * cfg.block_size, "params_quantized": N_Q,
          "params_total": N_ALL, "params_embedding": N_EMB, "rows": rows,
          "hist": {"range": round(hist_range, 5), "counts": [int(c) for c in hist.tolist()]},
          "outlier": outlier, "row_examples": examples, "per_layer_outlier": per_layer}
if args.json:
    with open(args.json, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
