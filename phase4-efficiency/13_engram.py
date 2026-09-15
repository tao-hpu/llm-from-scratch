"""
里程碑 13(第 15 章):N-gram 查表 —— 给模型挂一本"短语词典",容量涨、每步算力几乎不涨
===================================================================================

第 11 章的 MoE 做了一笔交易:总参数翻倍,每 token 激活的算力不变。
这一关把交易做得更极端:加一张很大的表,每个 token 只按"最近两三个字"去表里取几行。

  【零件 1】hash     把 (前一个字, 当前字) 和 (前两个字, 前一个字, 当前字) 各算出几个行号
  【零件 2】lookup   每个行号从表里取一行向量,拼成一根 n_embd 维的"记忆向量" e
  【零件 3】gate     用当前隐状态 h 和 e 算一个 0~1 的门:这段记忆和上下文对不对得上
  【零件 4】inject   门 × 记忆,过一个短卷积,加回残差流(只在第 2 层加一次)

表可以做得非常大,但每个 token 只读固定几行,和表多大无关;
而且行号只取决于输入 token,算 forward 之前就知道要读哪几行,
所以表可以放在 CPU 内存甚至 SSD 上,提前取好再送进 GPU。

出处:DeepSeek Engram(arXiv 2601.07372)、N-Grammer(arXiv 2207.06366);
Qwen3.8-Flash-Next 技术报告里的 N-gram Embedding 层也是这个做法。

跑法(默认读 FineWeb-Edu 验证片 edufineweb_val_000000.npy,见 find_val_shard;--data shakespeare 换成字符级反例):
    python 13_engram.py --table 0              # 不挂表(= 第 12 章四件齐上,对照组)
    python 13_engram.py --table 4096           # 每个哈希头 4096 行
    python 13_engram.py --table 32768          # 每个哈希头 32768 行(表 4.19M 参数,FineWeb 默认配置下主干 7.24M)
    python 13_engram.py --table 32768 --orders 2        # 只查 2-gram
    python 13_engram.py --table 32768 --table-device cpu # 表放 CPU,只把取出来的行送进 GPU
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- 超参数(与第 12 章一致)----
batch_size = 64
block_size = 64
n_embd = 128
n_head = 4
n_kv_head = 2
n_layer = 4
dropout = 0.1
head_size = n_embd // n_head

heads_per_order = 4   # 每种 n-gram 用几个哈希头(Engram 论文:8)
inject_layer = 1      # 在第几个 Block 前注入(0 起数;1 = 第 2 层,和 Engram / Qwen3.8 一致)

lr = 1e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200

p = argparse.ArgumentParser()
p.add_argument("--table", type=int, default=32768, help="每个哈希头的表行数;0 = 不挂表")
p.add_argument("--orders", type=str, default="2,3", help="查哪几阶 n-gram")
p.add_argument("--table-device", type=str, default="same", choices=["same", "cpu"],
               help="same = 表和模型放同一设备;cpu = 表留在 CPU,只搬运取出的行")
p.add_argument("--data", type=str, default="fineweb", choices=["fineweb", "shakespeare"],
               help="fineweb = GPT-2 BPE token(约 1 亿,只过一遍);shakespeare = 字符级 1M(数据少,表会背训练集)")
p.add_argument("--max-iters", type=int, default=None)
p.add_argument("--eval-iters", type=int, default=None)
p.add_argument("--json", type=str, default="")
args = p.parse_args()
TABLE = args.table
ORDERS = [int(o) for o in args.orders.split(",")] if TABLE > 0 else []
if args.max_iters is not None:
    max_iters = args.max_iters
if args.eval_iters is not None:
    eval_iters = args.eval_iters
TAG = "none" if TABLE == 0 else f"{'+'.join(map(str, ORDERS))}gram-{TABLE}"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)
print(f"[{TAG}] device = {device} | 表设备 = {args.table_device}")

HERE = os.path.dirname(os.path.abspath(__file__))

def find_val_shard():
    """FineWeb-Edu 验证片:先找本目录 data/,再找第 6 章 prepare_fineweb.py 的默认输出 phase1-124m/data/。"""
    for d in (os.path.join(HERE, "data"), os.path.join(HERE, "..", "phase1-124m", "data")):
        f = os.path.join(d, "edufineweb_val_000000.npy")
        if os.path.exists(f):
            return f
    raise SystemExit("找不到 edufineweb_val_000000.npy:先在 phase1-124m/ 下跑 prepare_fineweb.py,"
                     "或把这一片放进 phase4-efficiency/data/")

if args.data == "shakespeare":
    with open(os.path.join(HERE, "..", "phase1-nanogpt", "data", "tinyshakespeare.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    tok_strings = lambda ids: [itos[i] for i in ids]
else:
    # FineWeb-Edu 的一个 shard:GPT-2 BPE,约 1 亿 token(第 6 章 prepare_fineweb.py 的产物)。
    # 5000 步 × 4096 token ≈ 2050 万 token,远少于训练段,每段文本只见一次。
    # 记忆表要在这种"数据远多于参数"的条件下才有意义:tiny shakespeare 只有 1M 字符,表会把训练集背下来。
    import numpy as np
    import tiktoken
    arr = np.load(find_val_shard(), mmap_mode="r")
    data = torch.from_numpy(np.array(arr, dtype=np.int64))
    vocab_size, stoi = 50257, {}
    enc = tiktoken.get_encoding("gpt2")
    tok_strings = lambda ids: [enc.decode([i]) for i in ids]
    n = len(data) - 5_000_000
    dropout = 0.0                      # 数据只过一遍,不需要 dropout
    eval_iters = min(eval_iters, 50)
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    # 行号只取决于这个窗口里的 token(窗口外按 0 补),在 CPU 上算好随 batch 一起给出
    r = ngram_rows(x) if TABLE > 0 else None
    return x, y, r

# ===========================================================================
# 主干零件(同第 12 章)
# ===========================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

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

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * head_size, bias=False)
        self.o_proj = nn.Linear(n_head * head_size, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        cos, sin = build_rope_cache(block_size, head_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, n_head, head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, n_kv_head, head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, n_kv_head, head_size).transpose(1, 2)
        q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        rep = n_head // n_kv_head
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                                           is_causal=True, dropout_p=dropout if self.training else 0.0)
        return self.drop(self.o_proj(y.transpose(1, 2).reshape(B, T, C)))

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

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.ffn = Attention(), SwiGLU()
        self.n1, self.n2 = RMSNorm(n_embd), RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))

# ===========================================================================
# 本关主角:哈希 n-gram 查表
# ===========================================================================
N_HEADS = heads_per_order * len(ORDERS)
ROW_DIM = n_embd // max(N_HEADS, 1)
MEM_DIM = N_HEADS * ROW_DIM          # 查表拼出的维数,由 w_k / w_v 投影回 n_embd

def hash_multipliers():
    """每个哈希头的随机奇数乘子(固定种子,每次运行都一样;页面上的哈希演示用的就是这组数)。"""
    g = torch.Generator().manual_seed(2026)
    return [(torch.randint(1, 2**20, (order,), generator=g) * 2 + 1).tolist()
            for order in ORDERS for _ in range(heads_per_order)]

def ngram_rows(idx):
    """【零件 1】idx (B,T) → 行号 (B,T,N_HEADS)。只看输入 token,不需要任何模型计算。

    第 h 个头:把最近 order 个 token 各乘一个随机奇数,再异或到一起,对表长取模。
    不同的头用不同的乘数,两个 n-gram 在某个头上撞了行,在别的头上大概率撞不上。
    开头不足 order 个 token 的位置,用 0 号 token 补齐。"""
    B, T = idx.shape
    rows, k = [], 0
    for order in ORDERS:
        padded = F.pad(idx, (order - 1, 0))                                # 前面补 0
        grams = padded.unfold(1, order, 1)                                 # (B,T,order):以 t 结尾的 order 个 token
        for _ in range(heads_per_order):
            mult = MULTS[k]; k += 1                                        # 这个头的随机奇数乘子
            h = torch.zeros(B, T, dtype=torch.long)
            for j in range(order):
                h = h ^ (grams[..., j] * mult[j])                          # 乘法 + 异或
            rows.append(h % TABLE)
    return torch.stack(rows, dim=-1)

MULTS = [torch.tensor(m) for m in hash_multipliers()]

class NgramMemory(nn.Module):
    def __init__(self):
        super().__init__()
        # 【零件 2】N_HEADS 张表,每张 TABLE 行、每行 ROW_DIM 维;拼起来 MEM_DIM 维(n_embd 除不尽时略小于 n_embd)
        self.tables = nn.Parameter(torch.randn(N_HEADS, TABLE, ROW_DIM) * 0.02)
        # 【零件 3】门控:h 当 query,记忆当 key,点积过 sigmoid
        self.w_k = nn.Linear(MEM_DIM, n_embd, bias=False)
        self.w_v = nn.Linear(MEM_DIM, n_embd, bias=False)
        self.norm_h, self.norm_k = RMSNorm(n_embd), RMSNorm(n_embd)
        # 【零件 4】短卷积(kernel=4,因果)
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=4, groups=n_embd, padding=3, bias=False)
        self.last_gate = None

    def lookup(self, rows):
        """rows (B,T,N_HEADS) → e (B,T,MEM_DIM)。每个 token 只读 N_HEADS 行。"""
        head_ids = torch.arange(N_HEADS, device=self.tables.device)
        e = self.tables[head_ids, rows.to(self.tables.device)]            # (B,T,N_HEADS,ROW_DIM)
        return e.flatten(-2)

    def forward(self, h, e):
        B, T, C = h.shape
        gate = torch.sigmoid((self.norm_h(h) * self.norm_k(self.w_k(e))).sum(-1, keepdim=True) / C ** 0.5)
        self.last_gate = gate.detach()
        y = gate * self.w_v(e)
        return self.conv(y.transpose(1, 2))[..., :T].transpose(1, 2)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        if args.data == "fineweb":
            self.lm_head.weight = self.tok.weight          # 词表 5 万,嵌入和输出头共用一张表
            nn.init.normal_(self.tok.weight, std=0.02)     # 共用后按 GPT-2 的尺度初始化,否则初始 logits 过大
        self.mem = NgramMemory() if TABLE > 0 else None

    def forward(self, idx, targets=None, e=None):
        x = self.tok(idx)
        for i, b in enumerate(self.blocks):
            if i == inject_layer and self.mem is not None:
                x = x + self.mem(x, e)                    # 只注入一次
            x = b(x)
        logits = self.lm_head(self.norm_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

model = Model().to(device)
if model.mem is not None and args.table_device == "cpu":
    model.mem.tables.data = model.mem.tables.data.cpu()   # 大表留在 CPU,其余上 GPU

def run(x, y=None, rows=None):
    """按行号取出那几行(表在 CPU 时就在 CPU 上取),再把取出来的行送进设备。"""
    e = None
    if model.mem is not None:
        rows = ngram_rows(x) if rows is None else rows
        e = model.mem.lookup(rows).to(device)
    return model(x.to(device), None if y is None else y.to(device), e)

n_table = model.mem.tables.numel() if model.mem is not None else 0
n_param = sum(q.numel() for q in model.parameters())
n_backbone = n_param - n_table
print(f"[{TAG}] 主干参数 {n_backbone/1e6:.3f} M | 表参数 {n_table/1e6:.3f} M"
      f" | 每 token 读表 {N_HEADS} 行 × {ROW_DIM} 维 = {N_HEADS * ROW_DIM} 个数")

# ---- 撞行统计:训练集里出现过的 n-gram,有多少对在同一个头上共用了一行 ----
collision = {}
if TABLE > 0:
    sample = train_data[:200_000].view(1, -1)
    rows = ngram_rows(sample)[0]
    k = 0
    for order in ORDERS:
        grams = F.pad(sample, (order - 1, 0)).unfold(1, order, 1)[0]
        uniq, inv = torch.unique(grams, dim=0, return_inverse=True)
        first = torch.zeros(len(uniq), dtype=torch.long).scatter_(0, inv, torch.arange(len(inv)))
        for _ in range(heads_per_order):
            r = rows[first, k]
            collision[f"{order}gram_h{k}"] = round(1 - torch.unique(r).numel() / len(uniq), 4)
            k += 1
        collision[f"{order}gram_distinct"] = len(uniq)
    print(f"[{TAG}] 训练集前 20 万字里的不同 n-gram 与撞行比例: {collision}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            losses[i] = run(*get_batch(split))[1].item()
        out[split] = losses.mean().item()
    model.train()
    return out

curve = []
t0 = time.time()
tokens, step_seconds = 0, 0.0
print(f"===== 训练 [{TAG}] =====")
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        l = estimate_loss()
        curve.append([it, round(l["train"], 4), round(l["val"], 4)])
        print(f"step {it:4d} | train {l['train']:.4f} | val {l['val']:.4f} | {time.time()-t0:.0f}s")
    ts = time.time()
    _, loss = run(*get_batch("train"))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    step_seconds += time.time() - ts
    tokens += batch_size * block_size
train_seconds = time.time() - t0

result = {"tag": TAG, "table": TABLE, "orders": ORDERS, "heads": N_HEADS, "row_dim": ROW_DIM,
          "table_device": args.table_device, "params_backbone": n_backbone, "params_table": n_table,
          "curve": curve, "val": curve[-1][2], "train_seconds": round(train_seconds),
          "tokens_per_sec": round(tokens / step_seconds), "collision": collision,
          "multipliers": hash_multipliers() if TABLE > 0 else [], "stoi": stoi}

# ---- 门控长什么样:一段 val 文本,每个位置的门值 ----
if model.mem is not None:
    model.eval()
    with torch.no_grad():
        x = val_data[2000:2000 + block_size].view(1, -1)
        run(x)
        g = model.mem.last_gate[0, :, 0].cpu()
    result["gate_tokens"] = tok_strings(x[0].tolist())
    result["gate_values"] = [round(v, 3) for v in g.tolist()]
    print("门值(每个字符一格):")
    print("|".join(tok_strings(x[0].tolist())).replace("\n", "⏎"))
    print(" ".join(f"{v:.2f}" for v in g.tolist()))

if TABLE > 0:
    ex = val_data[2000:2024].view(1, -1)
    result["hash_example"] = {"tokens": tok_strings(ex[0].tolist()), "ids": ex[0].tolist(),
                              "rows": ngram_rows(ex)[0].tolist()}
result["data"] = args.data

print(f"[{TAG}] 完成 | val {result['val']:.4f} | {result['tokens_per_sec']} tok/s | 用时 {train_seconds:.0f}s")
if args.json:
    with open(args.json, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
