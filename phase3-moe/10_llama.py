"""
里程碑 10(第 12 章)：把 GPT-2 老架构升级成 LLaMA 同款
========================================================

第 3 章那个 GPT 是 2019 年的样子。2023 年之后的开源模型(LLaMA / Qwen /
Mistral / DeepSeek)长得都不太一样了,但**换掉的不是 Transformer 本身**,
而是它身上四个零件。这一关把四个零件一个一个换过去,每换一个都能单独开关:

  【零件 1】RoPE      位置嵌入表 → 旋转位置编码(把位置编进 q/k 的"角度"里)
  【零件 2】RMSNorm   LayerNorm  → 只除以均方根,不减均值(更少的算子)
  【零件 3】SwiGLU    ReLU 的 FFN → 门控 FFN(一路当"内容"、一路当"闸门")
  【零件 4】GQA       每头一份 K/V → 多个 query 头共享一份 K/V(推理省显存)

注意力/残差/训练循环/数据管线全都没动 —— 和第 11 章 MoE 一样,
这一关也是"只换零件,不换骨架"。

跑法(每个开关都能单独拨,方便做 A/B):
    python3 10_llama.py                      # 四件齐上(LLaMA 同款,默认)
    python3 10_llama.py --preset gpt2        # 全关 = 第 3 章那套老架构
    python3 10_llama.py --preset rope        # 只开 RoPE,其余保持老架构
    python3 10_llama.py --preset swiglu      # 只开 SwiGLU
    python3 10_llama.py --preset llama --n-kv-head 1   # 极端 GQA(=MQA)

预期:四件齐上 val loss 略优于第 3 章的 ~1.57,同时 KV cache 只有原来的 1/2。
     单看某一件的收益都不大,这一关想让你看到的是"现代架构 = 四件小改的叠加"。
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- 超参数(和第 3 章 03_transformer.py 完全一致,方便直接对照)----
batch_size = 64
block_size = 64       # 上下文长度
n_embd = 128          # 主干维度
n_head = 4            # query 头数
n_layer = 4           # 层数
dropout = 0.1

lr = 1e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200

# ---------------------------------------------------------------------------
# 四个开关
# ---------------------------------------------------------------------------
PRESETS = {   # 预设 = (rope, rmsnorm, swiglu, n_kv_head)
    "gpt2":    (0, 0, 0, n_head),   # 第 3 章那套老架构
    "rope":    (1, 0, 0, n_head),
    "rmsnorm": (0, 1, 0, n_head),
    "swiglu":  (0, 0, 1, n_head),
    "gqa":     (0, 0, 0, 2),
    "llama":   (1, 1, 1, 2),        # 四件齐上
}

p = argparse.ArgumentParser()
p.add_argument("--preset", type=str, default="llama", choices=list(PRESETS),
               help="一次拨好四个开关;想单独控制就用下面四个参数覆盖")
p.add_argument("--rope", type=int, default=None, help="1=RoPE, 0=可学习位置嵌入表")
p.add_argument("--rmsnorm", type=int, default=None, help="1=RMSNorm, 0=LayerNorm")
p.add_argument("--swiglu", type=int, default=None, help="1=SwiGLU, 0=ReLU FFN")
p.add_argument("--n-kv-head", type=int, default=None,
               help="K/V 头数:=n_head 就是普通 MHA;<n_head 是 GQA;=1 是 MQA")
p.add_argument("--max-iters", type=int, default=max_iters)
p.add_argument("--tag", type=str, default="", help="打印用的标签,方便多组对照")
p.add_argument("--optim", type=str, default="adamw", choices=["adamw", "muon"],
               help="muon = 隐藏层的 2D 权重用 Muon,嵌入/输出头/归一化仍用 AdamW")
p.add_argument("--muon-lr", type=float, default=0.02)
p.add_argument("--json", type=str, default="", help="把 loss 曲线写进这个 json 文件")
args = p.parse_args()

USE_ROPE, USE_RMS, USE_SWIGLU, N_KV_HEAD = PRESETS[args.preset]
if args.rope is not None:       USE_ROPE = args.rope
if args.rmsnorm is not None:    USE_RMS = args.rmsnorm
if args.swiglu is not None:     USE_SWIGLU = args.swiglu
if args.n_kv_head is not None:  N_KV_HEAD = args.n_kv_head
max_iters = args.max_iters

head_size = n_embd // n_head
assert n_head % N_KV_HEAD == 0, "n_head 必须能被 n_kv_head 整除(每组共享一份 K/V)"
N_REP = n_head // N_KV_HEAD          # 每份 K/V 被几个 query 头共用

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)

name = args.tag or args.preset
print(f"[{name}] device = {device} | RoPE={USE_ROPE} RMSNorm={USE_RMS} "
      f"SwiGLU={USE_SWIGLU} n_kv_head={N_KV_HEAD}(n_head={n_head})")

# ---- 数据(还是 tiny shakespeare,复用 phase1 的文件)----
# 路径按脚本所在目录算,不看你从哪儿敲的命令。
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "phase1-nanogpt", "data", "tinyshakespeare.txt")
with open(DATA, "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ===========================================================================
# 【零件 2】RMSNorm:比 LayerNorm 少做一半的事
# ===========================================================================
class RMSNorm(nn.Module):
    """LayerNorm 做两件事:①减均值(把这根向量挪到 0 附近) ②除以标准差。
    RMSNorm 只做第 ②:除以"均方根"。少一次求均值、少一个 bias 参数,
    实测效果基本持平 —— 所以现在的模型基本都换成了它。"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 只有缩放,没有 bias

    def forward(self, x):
        # rsqrt(mean(x²)) 就是"除以均方根"
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

def make_norm(dim):
    return RMSNorm(dim) if USE_RMS else nn.LayerNorm(dim)

# ===========================================================================
# 【零件 1】RoPE:把位置编进 q/k 的"角度"里
# ===========================================================================
def build_rope_cache(seq_len, dim, base=10000.0):
    """预先算好每个位置、每个频率对应的 cos/sin。
    dim 是每个头的维度,必须是偶数(两两一对当成复平面上的一个点来转)。"""
    assert dim % 2 == 0
    # 每一对维度配一个频率:越靠后的对转得越慢(波长越长 → 管长距离)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))   # (dim/2,)
    t = torch.arange(seq_len).float()                                    # (T,)
    freqs = torch.outer(t, inv_freq)                                     # (T, dim/2) 角度 = 位置 × 频率
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    """x: (B, n_head, T, head_size)。把每两个相邻维度当成一个二维向量,
    按"这个位置该转多少度"整体旋转。位置信息就藏在旋转角里。"""
    T = x.size(2)
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)     # (T, hs/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]                 # 偶数位 / 奇数位
    # 标准二维旋转:[x1, x2] → [x1·cos − x2·sin, x1·sin + x2·cos]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2)    # 交错着拼回去

# ===========================================================================
# 注意力:一次算完所有头(比第 3 章的 ModuleList 写法更接近真实实现)
# 【零件 4】GQA 就体现在 K/V 的头数上
# ===========================================================================
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # query 有 n_head 份;key/value 只有 n_kv_head 份 —— 这就是 GQA 省的地方
        self.q_proj = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, N_KV_HEAD * head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, N_KV_HEAD * head_size, bias=False)
        self.o_proj = nn.Linear(n_head * head_size, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        if USE_ROPE:
            cos, sin = build_rope_cache(block_size, head_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, n_head,   head_size).transpose(1, 2)  # (B,nh,T,hs)
        k = self.k_proj(x).view(B, T, N_KV_HEAD, head_size).transpose(1, 2) # (B,nkv,T,hs)
        v = self.v_proj(x).view(B, T, N_KV_HEAD, head_size).transpose(1, 2)

        if USE_ROPE:                       # 【零件 1】位置在这里进入模型
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        if N_REP > 1:                      # 【零件 4】一份 K/V 复制给同组的几个 query 头
            k = k.repeat_interleave(N_REP, dim=1)
            v = v.repeat_interleave(N_REP, dim=1)

        # 因果注意力(和前几章同一个公式,这里交给 PyTorch 的融合实现)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, n_head * head_size)
        return self.drop(self.o_proj(y))

# ===========================================================================
# 【零件 3】FFN:ReLU 版 vs SwiGLU 版
# ===========================================================================
class FeedForward(nn.Module):
    """第 3 章那版:放大 4 倍 → ReLU → 压回。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class SwiGLU(nn.Module):
    """门控 FFN:同一个输入走两路,一路过 SiLU 当"闸门",一路当"内容",逐元素相乘。
    闸门决定每个通道放多少内容过去,比"一刀切负数"的 ReLU 精细。
    代价是多了一个矩阵(三个而不是两个),所以隐层要收窄到 8/3·n_embd 才与 ReLU 版参数量相当。"""
    def __init__(self):
        super().__init__()
        hidden = int(8 * n_embd / 3)
        hidden = (hidden + 7) // 8 * 8          # 凑成 8 的倍数,对硬件友好
        self.gate = nn.Linear(n_embd, hidden, bias=False)   # 闸门那一路
        self.up   = nn.Linear(n_embd, hidden, bias=False)   # 内容那一路
        self.down = nn.Linear(hidden, n_embd, bias=False)   # 压回主干
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

def make_ffn():
    return SwiGLU() if USE_SWIGLU else FeedForward()

# ---- Block:和第 3 章一样的 pre-norm 残差,只是零件换了 ----
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.ffn = make_ffn()
        self.n1 = make_norm(n_embd)
        self.n2 = make_norm(n_embd)
    def forward(self, x):
        x = x + self.attn(self.n1(x))     # 沟通
        x = x + self.ffn(self.n2(x))      # 思考
        return x

class LlamaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 用 RoPE 就不需要这张可学习的位置表了(位置在注意力内部进入模型)
        self.position_embedding_table = None if USE_ROPE else nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.norm_f = make_norm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        if self.position_embedding_table is not None:
            x = x + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx

model = LlamaGPT().to(device)
n_param = sum(p_.numel() for p_ in model.parameters())
# 推理时每个 token 要缓存的 K/V 数值个数(GQA 省的就是这个)
kv_per_token = 2 * n_layer * N_KV_HEAD * head_size
kv_dense     = 2 * n_layer * n_head * head_size
print(f"[{name}] 总参数 = {n_param/1e6:.3f} M | 每 token 的 KV cache = {kv_per_token} 个数"
      f"(MHA 是 {kv_dense},省 {100*(1-kv_per_token/kv_dense):.0f}%)")

# ---- 训练(和第 3 章同一个循环)----
# ===========================================================================
# 可选:Muon 优化器(Keller Jordan 2024)
# AdamW 对每个数各自调步长;Muon 把一个 2D 权重的更新当成整体,先做动量,
# 再用 Newton-Schulz 迭代把更新矩阵"正交化"(奇异值都推到 1 附近),
# 这样每个方向走的步子差不多大,不会被少数几个大奇异值方向主导。
# ===========================================================================
def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)      # 五次多项式系数,出自 Keller Jordan 的博客
    X = G / (G.norm() + eps)
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p_ in g["params"]:
                if p_.grad is None:
                    continue
                buf = self.state[p_].setdefault("buf", torch.zeros_like(p_))
                buf.mul_(g["momentum"]).add_(p_.grad)
                u = newton_schulz(p_.grad.add(buf, alpha=g["momentum"]))     # Nesterov 动量后正交化
                p_.add_(u, alpha=-g["lr"] * max(1, p_.size(0) / p_.size(1)) ** 0.5)

if args.optim == "muon":
    hidden_2d = [q for n_, q in model.named_parameters() if n_.startswith("blocks.") and q.dim() == 2]
    others = [q for n_, q in model.named_parameters() if not (n_.startswith("blocks.") and q.dim() == 2)]
    muon = Muon(hidden_2d, lr=args.muon_lr)
    adamw = torch.optim.AdamW(others, lr=lr)

    class Both:
        def zero_grad(self, set_to_none=True):
            muon.zero_grad(set_to_none=set_to_none); adamw.zero_grad(set_to_none=set_to_none)

        def step(self):
            muon.step(); adamw.step()
    optimizer = Both()
    print(f"[{name}] Muon 管 {sum(q.numel() for q in hidden_2d)/1e6:.3f} M 个隐藏层权重,"
          f"AdamW 管其余 {sum(q.numel() for q in others)/1e6:.3f} M")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

print(f"===== 训练 [{name}] =====")
t_start = time.time()
tokens_done = 0
curve = []
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        l = estimate_loss()
        el = time.time() - t_start
        curve.append([it, round(l["train"], 4), round(l["val"], 4)])
        print(f"step {it:4d} | train loss {l['train']:.4f} | val loss {l['val']:.4f}"
              f" | {tokens_done/max(el,1e-9):,.0f} tok/s")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    tokens_done += batch_size * block_size

final = estimate_loss()
print(f"[{name}] 训练结束 | val loss {final['val']:.4f} | 参数 {n_param/1e6:.3f} M "
      f"| KV/token {kv_per_token} | 用时 {time.time()-t_start:.0f}s")
if args.json:
    import json
    with open(args.json, "w") as f:
        json.dump({"name": name, "optim": args.optim, "curve": curve, "val": round(final["val"], 4),
                   "params": n_param, "seconds": round(time.time() - t_start)}, f, ensure_ascii=False, indent=1)

print(f"\n----- 采样结果([{name}],生成 400 字)-----")
model.eval()   # 生成时关掉 dropout(新版 PyTorch 的 MPS 后端也不支持带 dropout 的 SDPA)
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, max_new_tokens=400)[0].tolist()))
