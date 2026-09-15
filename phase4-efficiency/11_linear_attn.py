"""
里程碑 11(第 13 章):线性注意力 —— 把越长越大的 KV cache 换成一块固定大小的"记忆板"
================================================================================

第 12 章的注意力有一笔甩不掉的账:推理时每个 token 的 K/V 都要存着,上下文越长 cache 越大,
每生成一个新 token 还要和全部历史逐个打分。线性注意力换了个思路:

  【思路】不存历史 K/V,只维护一块固定大小的矩阵 S(形状 head_size × head_size)。
          每来一个 token,就把 (k, v) 这对"写"进 S;要读时拿 q 去乘 S。
          S 的大小和上下文长度无关 —— 模型变回了一个 RNN。

同一块记忆板,有三种写法,也是这一关的三个开关:

  linear   S ← S + v·kᵀ                         只加不删,写多了互相覆盖(Katharopoulos 2020)
  delta    S ← S + β·(v − S·k)·kᵀ               先读出 k 对应的旧值,只补差额(delta rule)
  gdn      S ← α·S + β·(v − α·S·k)·kᵀ           再加一个遗忘门 α,整块记忆按比例衰减
                                                 (Gated DeltaNet,Qwen3-Next 的主力层)

记忆板有容量上限,精确检索("前面第 37 个 key 对应的 value 是什么")会吃亏。
所以真实模型不全换:每 4 层留 1 层真注意力(hybrid,3:1)。

跑法:
    python 11_linear_attn.py --preset attn       # 4 层全是注意力(对照组,= 第 12 章四件齐上)
    python 11_linear_attn.py --preset linear     # 4 层 linear
    python 11_linear_attn.py --preset delta      # 4 层 delta rule
    python 11_linear_attn.py --preset gdn        # 4 层 Gated DeltaNet
    python 11_linear_attn.py --preset hybrid     # 3 层 GDN + 1 层注意力
    python 11_linear_attn.py --preset gdn --task recall   # 换成"键值检索"任务,看记忆板的容量上限

任务 lm:tiny shakespeare 字符级语言模型,训练长度 64;训完再在 64/128/256/512 长度上测 val loss。
任务 recall:序列前半段是 N 对 (key, value),后半段把 key 打乱再问一遍,模型要答出对应 value。
            N 越大,记忆板要同时记住的东西越多。
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- 超参数(与第 12 章 10_llama.py 一致)----
batch_size = 64
block_size = 64
n_embd = 128
n_head = 4
n_kv_head = 2
n_layer = 4
dropout = 0.1
head_size = n_embd // n_head

lr = 1e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200

PRESETS = {
    "attn":   ["attn", "attn", "attn", "attn"],
    "linear": ["linear", "linear", "linear", "linear"],
    "delta":  ["delta", "delta", "delta", "delta"],
    "gdn":    ["gdn", "gdn", "gdn", "gdn"],
    "hybrid": ["gdn", "gdn", "gdn", "attn"],   # 3:1,最后一层留给真注意力
}

p = argparse.ArgumentParser()
p.add_argument("--preset", type=str, default="hybrid", choices=list(PRESETS))
p.add_argument("--task", type=str, default="lm", choices=["lm", "recall"])
p.add_argument("--max-iters", type=int, default=None)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--batch", type=int, default=None)
p.add_argument("--recall-ns", type=str, default="8,16,24,32", help="recall 任务评测时 key/value 对数的几档")
p.add_argument("--train-n", type=int, default=32, help="recall 任务训练时固定的 key/value 对数"
               "(实测训练时混着采样不同 N,全注意力 6000 步也学不会;固定 N 才会出现 loss 骤降)")
p.add_argument("--eval-iters", type=int, default=None, help="每次估 loss 抽几个 batch(冒烟测试时调小)")
p.add_argument("--eval-lens", type=str, default="64,128,256,512",
               help="lm 任务训完后,在这些上下文长度上测 val loss(训练只见过 64)")
p.add_argument("--json", type=str, default="", help="把关键数字写进这个 json 文件")
args = p.parse_args()

LAYERS = PRESETS[args.preset]
TASK = args.task
if TASK == "recall":
    # 检索任务要先经过一段平台期,模型学会"找到同一个 key、抄它后面那个 token"之后 loss 才骤降;
    # 实测 lr 1e-3、batch 128 约 2500–3000 步出现这个跳变(lr 3e-3 不稳)
    max_iters, eval_interval, batch_size = 6000, 500, 128
if args.max_iters is not None:
    max_iters = args.max_iters
if args.eval_iters is not None:
    eval_iters = args.eval_iters
if args.lr is not None:
    lr = args.lr
if args.batch is not None:
    batch_size = args.batch
if args.task == "recall":
    dropout = 0.0   # 合成任务数据无限,不会过拟合,dropout 只会拖慢学会检索
EVAL_LENS = [int(x) for x in args.eval_lens.split(",")] if TASK == "lm" else []
RECALL_NS = [int(x) for x in args.recall_ns.split(",")]
MAX_T = max([block_size] + EVAL_LENS + ([4 * max(RECALL_NS)] if TASK == "recall" else []))

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)
print(f"[{args.preset}/{TASK}] device = {device} | 每层 = {LAYERS}")

# ===========================================================================
# 数据
# ===========================================================================
if TASK == "lm":
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "phase1-nanogpt", "data", "tinyshakespeare.txt")
    with open(DATA, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split, T=block_size):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - T, (batch_size,))
        x = torch.stack([d[i:i + T] for i in ix])
        y = torch.stack([d[i + 1:i + T + 1] for i in ix])
        return x.to(device), y.to(device)
else:
    # 键值检索(multi-query associative recall 的简化版)
    #   前半段: k1 v1 k2 v2 ... kN vN
    #   后半段: 把 N 个 key 打乱再出一遍,每个 key 后面那个位置要预测出它的 value
    N_KEYS, N_VALS = 64, 64          # key 用 0..63,value 用 64..127
    vocab_size = N_KEYS + N_VALS

    def get_batch(split, N=None):
        N = N or args.train_n
        keys = torch.stack([torch.randperm(N_KEYS)[:N] for _ in range(batch_size)])
        vals = torch.randint(N_KEYS, N_KEYS + N_VALS, (batch_size, N))
        order = torch.stack([torch.randperm(N) for _ in range(batch_size)])
        qk = torch.gather(keys, 1, order)
        qv = torch.gather(vals, 1, order)
        ctx = torch.stack([keys, vals], dim=2).view(batch_size, 2 * N)
        qry = torch.stack([qk, qv], dim=2).view(batch_size, 2 * N)
        seq = torch.cat([ctx, qry], dim=1)                        # (B, 4N)
        x, y = seq[:, :-1], seq[:, 1:].clone()
        # 只在"刚看到查询 key、下一个该答 value"的位置算 loss,其余位置忽略
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, 2 * N::2] = True                                  # y 的这些位置是 value
        y[~mask] = -100
        return x.to(device), y.to(device)

# ===========================================================================
# 公用零件(直接取自第 12 章:RMSNorm / RoPE / SwiGLU)
# ===========================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

def build_rope_cache(seq_len, dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    T = x.size(2)
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = (int(8 * n_embd / 3) + 7) // 8 * 8
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

# ---- 对照组:第 12 章那层注意力(RoPE + GQA),推理时要存 KV cache ----
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * head_size, bias=False)
        self.o_proj = nn.Linear(n_head * head_size, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        cos, sin = build_rope_cache(MAX_T, head_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, n_head, head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, n_kv_head, head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, n_kv_head, head_size).transpose(1, 2)
        q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        rep = n_head // n_kv_head
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=dropout if self.training else 0.0)
        return self.drop(self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C)))

# ===========================================================================
# 本关主角:记忆板 S 的三种写法
# ===========================================================================
class RecurrentMixer(nn.Module):
    """一层"线性注意力"。和 Attention 接口一样:(B,T,C) 进,(B,T,C) 出。

    每个头维护一块 head_size × head_size 的记忆板 S。逐 token 做两件事:
      写:把 (k_t, v_t) 按 mode 规定的规则写进 S
      读:o_t = S_t · q_t
    这里用 for 循环逐 token 写,就是为了让"它是个 RNN"一眼看得出来。
    真实实现(flash-linear-attention)用分块并行算法,数学上等价,只是快得多。"""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        d = n_head * head_size
        self.q_proj = nn.Linear(n_embd, d, bias=False)
        self.k_proj = nn.Linear(n_embd, d, bias=False)
        self.v_proj = nn.Linear(n_embd, d, bias=False)
        # 短卷积(kernel=4,因果):让 q/k/v 先看到身边几个 token。GDN / Qwen3-Next 都带它;
        # 三种写法一视同仁地加上,对照时差别就只剩"写规则"。
        self.conv = nn.Conv1d(3 * d, 3 * d, kernel_size=4, groups=3 * d, padding=3, bias=False)
        if mode in ("delta", "gdn"):
            self.b_proj = nn.Linear(n_embd, n_head)          # β:这一步写多重(每头一个数)
        if mode == "gdn":
            self.a_proj = nn.Linear(n_embd, n_head)          # α:整块记忆保留几成
            nn.init.constant_(self.a_proj.bias, 3.0)         # 初始 α≈0.95,先别忘太快
        self.g_proj = nn.Linear(n_embd, d, bias=False)        # 输出门
        self.o_norm = RMSNorm(head_size)
        self.o_proj = nn.Linear(d, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_state=False):
        B, T, C = x.shape
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        qkv = F.silu(self.conv(qkv.transpose(1, 2))[..., :T].transpose(1, 2))
        q, k, v = qkv.view(B, T, 3, n_head, head_size).unbind(dim=2)   # 各 (B,T,nh,hs)

        if self.mode == "linear":
            # 原版线性注意力:特征映射保证非负,S 只加不减
            q, k = F.elu(q) + 1, F.elu(k) + 1
        else:
            # delta rule 要求 k 是单位向量:"读出 S·k"才等于"这个 key 当前存的值"
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        if self.mode in ("delta", "gdn"):
            beta = torch.sigmoid(self.b_proj(x))              # (B,T,nh) ∈ (0,1)
        if self.mode == "gdn":
            alpha = torch.sigmoid(self.a_proj(x))             # (B,T,nh) ∈ (0,1)

        S = x.new_zeros(B, n_head, head_size, head_size)      # 记忆板:大小与 T 无关
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]            # (B,nh,hs)
            if self.mode == "linear":
                S = S + vt.unsqueeze(-1) * kt.unsqueeze(-2)   # S ← S + v kᵀ
            else:
                if self.mode == "gdn":
                    S = alpha[:, t, :, None, None] * S        # 先整体遗忘:S ← α S
                v_old = (S @ kt.unsqueeze(-1)).squeeze(-1)    # 读出这个 key 现在存着什么
                b = beta[:, t, :, None]
                S = S + (b * (vt - v_old)).unsqueeze(-1) * kt.unsqueeze(-2)   # 只补差额
            outs.append((S @ qt.unsqueeze(-1)).squeeze(-1))   # 读:o = S q
        o = torch.stack(outs, dim=1)                          # (B,T,nh,hs)
        o = self.o_norm(o).reshape(B, T, -1) * F.silu(self.g_proj(x))
        out = self.drop(self.o_proj(o))
        return (out, S) if return_state else out

def make_mixer(kind):
    return Attention() if kind == "attn" else RecurrentMixer(kind)

class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.mix = make_mixer(kind)
        self.ffn = SwiGLU()
        self.n1, self.n2 = RMSNorm(n_embd), RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ffn(self.n2(x))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(kind) for kind in LAYERS])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.norm_f(self.blocks(self.tok(idx)))
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

model = Model().to(device)
n_param = sum(q.numel() for q in model.parameters())

# ---- 推理显存账:注意力层按 token 线性涨,记忆板层是常数 ----
n_attn = LAYERS.count("attn")
n_rec = n_layer - n_attn
kv_per_token = 2 * n_kv_head * head_size * n_attn             # 注意力层:每个 token 要存的数
state_const = n_head * head_size * head_size * n_rec          # 记忆板层:不管多长都是这么多
print(f"[{args.preset}] 参数 {n_param/1e6:.3f} M | 注意力层 {n_attn} 个,每 token KV {kv_per_token} 个数"
      f" | 记忆板层 {n_rec} 个,固定状态 {state_const} 个数")

# ===========================================================================
# 训练
# ===========================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss(iters=eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(iters)
        for i in range(iters):
            _, loss = model(*get_batch(split))
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

curve = []
t0 = time.time()
print(f"===== 训练 [{args.preset}/{TASK}] =====")
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        l = estimate_loss(eval_iters if TASK == "lm" else min(eval_iters, 50))
        curve.append([it, round(l["train"], 4), round(l["val"], 4)])
        print(f"step {it:4d} | train {l['train']:.4f} | val {l['val']:.4f} | {time.time()-t0:.0f}s")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
train_seconds = time.time() - t0

result = {"preset": args.preset, "task": TASK, "layers": LAYERS, "params": n_param,
          "kv_per_token": kv_per_token, "state_const": state_const,
          "curve": curve, "train_seconds": round(train_seconds)}

model.eval()
if TASK == "lm":
    result["val"] = curve[-1][2]
    # ---- 长度外推:训练只见过 64,拿更长的片段来测 ----
    by_len = {}
    with torch.no_grad():
        saved_bs = batch_size
        for T in EVAL_LENS:
            batch_size = max(4, saved_bs * block_size // T)       # 总 token 数大致不变
            ls = [model(*get_batch("val", T))[1].item() for _ in range(40)]
            by_len[T] = round(sum(ls) / len(ls), 4)
            print(f"  上下文 {T:4d}: val loss {by_len[T]:.4f}")
        batch_size = saved_bs
    result["val_by_len"] = by_len

    # ---- 记忆板长什么样:取第一个记忆板层,喂一段真实文本,看 S 的大小是否随长度变化 ----
    rec_blocks = [b for b in model.blocks if b.kind != "attn"]
    if rec_blocks:
        x, _ = get_batch("val", 256)
        h = model.tok(x[:1])
        for b in model.blocks:
            if b is rec_blocks[0]:
                _, S = b.mix(b.n1(h), return_state=True)
                break
            h = b(h)
        result["state_shape"] = list(S.shape)
        result["state_head0"] = [[round(v, 3) for v in row] for row in S[0, 0, :8, :8].tolist()]
    if rec_blocks and LAYERS[0] == "gdn":
        with torch.no_grad():
            x, _ = get_batch("val", 256)
            a = torch.sigmoid(rec_blocks[0].mix.a_proj(model.blocks[0].n1(model.tok(x))))
            result["alpha_mean_per_head"] = [round(v, 4) for v in a.mean(dim=(0, 1)).tolist()]
else:
    acc = {}
    with torch.no_grad():
        for N in RECALL_NS:
            hit = tot = 0
            for _ in range(20):
                x, y = get_batch("val", N)
                logits, _ = model(x)
                m = y != -100
                hit += (logits.argmax(-1)[m] == y[m]).sum().item()
                tot += m.sum().item()
            acc[N] = round(hit / tot, 4)
            print(f"  N = {N:2d} 对 key/value: 答对 {acc[N]*100:5.1f}%")
    result["recall_acc"] = acc

print(f"[{args.preset}/{TASK}] 完成 | 用时 {train_seconds:.0f}s")
if args.json:
    with open(args.json, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
