"""
里程碑 12(第 14 章):稀疏注意力 —— 不是每个 token 都要和全部历史打分
====================================================================

第 13 章把 KV cache 压成一块固定大小的记忆板,代价是精确检索变弱。
这一关走另一条路:K/V 照样全存,但每个 query 只挑一小部分去算。
挑法取自 DeepSeek 的 NSA(Native Sparse Attention, 2025),三条支路并行:

  【支路 1】compress  压缩:把历史按块(block_len 个 token)压成一个"摘要 key/value",query 先扫一遍摘要
  【支路 2】select    挑选:用支路 1 的打分,挑出 top_n 个最相关的块,块里的原始 token 逐个细看
  【支路 3】window    滑窗:最近 window 个 token 必看(局部语法离不开它)
  三条支路的输出用一个门控加权相加:out = g_cmp·o_cmp + g_sel·o_sel + g_win·o_win

每个 query 实际读的 token 数 ≈ window + top_n·block_len + (已过去的块数),
远小于"全部历史"。上下文越长,省得越多。

跑法:
    python 12_sparse_attn.py --branches full            # 对照组:普通因果注意力
    python 12_sparse_attn.py --branches win             # 只有滑窗
    python 12_sparse_attn.py --branches cmp,sel         # 压缩 + 挑选,不带滑窗
    python 12_sparse_attn.py --branches cmp,sel,win     # NSA 三条支路齐上(默认)
    python 12_sparse_attn.py --task recall              # 换成"远处找钥匙"任务:key/value 藏在滑窗够不到的地方

教学版用 mask 在稠密矩阵上"模拟"稀疏:算出来的数和真稀疏实现一致,但不会真的变快。
真正省时间要靠专门的 kernel(只取被选中的块),那是工程问题,不改变这里的算法。
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- 超参数:上下文拉到 256,稀疏才有意义 ----
batch_size = 32
block_size = 256
n_embd = 128
n_head = 4
n_kv_head = 2
n_layer = 4
dropout = 0.1
head_size = n_embd // n_head

block_len = 8      # 压缩 / 挑选的块长(NSA 论文:压缩块 32、步长 16、挑选块 64)
top_n = 4          # 每个 query 挑几块细看(含当前所在块)
window = 32        # 滑窗长度(NSA 论文:512)

lr = 1e-3
max_iters = 4000
eval_interval = 500
eval_iters = 100

p = argparse.ArgumentParser()
p.add_argument("--branches", type=str, default="cmp,sel,win",
               help="full = 普通注意力;否则从 cmp / sel / win 里挑,逗号分隔")
p.add_argument("--task", type=str, default="lm", choices=["lm", "recall"])
p.add_argument("--max-iters", type=int, default=None)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--eval-iters", type=int, default=None)
p.add_argument("--json", type=str, default="")
args = p.parse_args()

BRANCHES = args.branches.split(",")
FULL = BRANCHES == ["full"]
assert FULL or set(BRANCHES) <= {"cmp", "sel", "win"}
assert block_size % block_len == 0
if args.task == "recall":
    max_iters, batch_size = 6000, 128  # 检索要先熬过平台期(ch13 实测 lr 1e-3、batch 128 约 3000 步骤降)
if args.max_iters is not None:
    max_iters = args.max_iters
if args.eval_iters is not None:
    eval_iters = args.eval_iters
if args.lr is not None:
    lr = args.lr
if args.task == "recall":
    dropout = 0.0   # 合成任务数据无限,不会过拟合,dropout 只会拖慢学会检索
TAG = "full" if FULL else "+".join(BRANCHES)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)
print(f"[{TAG}/{args.task}] device = {device} | block_len={block_len} top_n={top_n} window={window}")

# ===========================================================================
# 数据
# ===========================================================================
if args.task == "lm":
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "phase1-nanogpt", "data", "tinyshakespeare.txt")
    with open(DATA, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device), None
else:
    # 远处找钥匙(128 个位置):
    #   位置 0          1 个噪声
    #   位置 1–32       16 对 (key, value)
    #   位置 33–96      64 个噪声
    #   位置 97–127     把 16 个 key 打乱再问一遍,每个 key 的下一个位置要答出它的 value
    # 滑窗(32)在提问区往回看,最远只到位置 66,够不到钥匙;要答对只能靠挑选支路把前面的块找回来。
    N_PAIRS, N_KEYS, N_VALS, N_FILL = 16, 32, 32, 32
    vocab_size = N_KEYS + N_VALS + N_FILL           # key 0..31 / value 32..63 / 噪声 64..95
    T_REC = 128
    Q0 = T_REC + 1 - 2 * N_PAIRS                    # 提问区在 x 里的起点(=97)

    def get_batch(split):
        B, P = batch_size, N_PAIRS
        seq = torch.randint(N_KEYS + N_VALS, vocab_size, (B, T_REC + 1))   # 先全部填噪声
        where = []
        for b in range(B):
            keys = torch.randperm(N_KEYS)[:P]
            vals = torch.randint(N_KEYS, N_KEYS + N_VALS, (P,))
            seq[b, 1:1 + 2 * P:2], seq[b, 2:2 + 2 * P:2] = keys, vals
            order = torch.randperm(P)
            seq[b, Q0::2], seq[b, Q0 + 1::2] = keys[order], vals[order]
            where.append((1 + 2 * order).tolist())                 # 第 i 个提问对应的钥匙位置
        x, y = seq[:, :-1], seq[:, 1:].clone()
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, Q0::2] = True                                       # 这些位置的下一个 token 是答案
        y[~mask] = -100
        return x.to(device), y.to(device), where

# ===========================================================================
# 零件(RMSNorm / RoPE / SwiGLU 同第 12 章)
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
    T = x.size(-2)
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

# ===========================================================================
# 本关主角:NSA 式稀疏注意力
# ===========================================================================
T_MAX = block_size
N_BLOCKS = T_MAX // block_len
pos = torch.arange(T_MAX)
CAUSAL = pos[None, :] <= pos[:, None]                                  # (T,T) 只看过去
WINDOW = CAUSAL & (pos[:, None] - pos[None, :] < window)               # (T,T) 只看最近 window 个
BLOCK_END = (torch.arange(N_BLOCKS) + 1) * block_len - 1               # 每块最后一个 token 的位置
BLOCK_DONE = BLOCK_END[None, :] <= pos[:, None]                        # (T,nb) 这块对 query t 已经完整
CUR_BLOCK = pos // block_len                                           # query t 所在的块

class SparseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(n_embd, n_head * head_size, bias=False)
        # 三条支路各有自己的 K/V 投影(NSA 的做法:防止一条支路的梯度"抄近道"影响另一条)
        names = ["full"] if FULL else BRANCHES
        self.kv = nn.ModuleDict({b: nn.Linear(n_embd, 2 * n_kv_head * head_size, bias=False) for b in names})
        if "cmp" in names or "sel" in names:
            # 压缩器:一块 block_len 个 key/value(带块内位置)→ 1 个摘要 key/value
            self.blk_pos = nn.Parameter(torch.zeros(block_len, head_size))
            self.cmp_k = nn.Linear(block_len * head_size, head_size, bias=False)
            self.cmp_v = nn.Linear(block_len * head_size, head_size, bias=False)
            # 一个永远可见的"空槽",免得开头几个 query 一块摘要都没有
            self.sink_kv = nn.Parameter(torch.zeros(2, n_kv_head, 1, head_size))
        if not FULL:
            self.gate = nn.Linear(n_embd, 3 * n_head)       # 每个头、每条支路一个门
        self.o_proj = nn.Linear(n_head * head_size, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        cos, sin = build_rope_cache(T_MAX, head_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.last = {}                                       # 留给可视化:门控值、挑中的块

    def _kv(self, branch, x):
        B, T, _ = x.shape
        k, v = self.kv[branch](x).view(B, T, 2, n_kv_head, head_size).unbind(dim=2)
        return k.transpose(1, 2), v.transpose(1, 2)          # (B,nkv,T,hs)

    def _attend(self, q, k, v, mask):
        rep = n_head // n_kv_head
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                              dropout_p=dropout if self.training else 0.0)

    def forward(self, x):
        B, T, C = x.shape
        dev = x.device
        q = self.q_proj(x).view(B, T, n_head, head_size).transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)

        if FULL:
            k, v = self._kv("full", x)
            y = self._attend(q, apply_rope(k, self.rope_cos, self.rope_sin), v, CAUSAL[:T, :T].to(dev))
            return self.drop(self.o_proj(y.transpose(1, 2).reshape(B, T, C)))

        outs = {}
        nb = T // block_len
        if "cmp" in BRANCHES or "sel" in BRANCHES:
            # ---- 支路 1:压缩 ----
            k, v = self._kv("cmp" if "cmp" in BRANCHES else "sel", x)
            kb = (k + self.blk_pos.repeat(nb, 1)[:T]).reshape(B, n_kv_head, nb, block_len * head_size)
            vb = (v + self.blk_pos.repeat(nb, 1)[:T]).reshape(B, n_kv_head, nb, block_len * head_size)
            ck, cv = self.cmp_k(kb), self.cmp_v(vb)                      # (B,nkv,nb,hs) 每块一个摘要
            # query t 只能看"已经完整结束"的块
            done = BLOCK_DONE[:T, :nb].to(dev)
            sink_k = self.sink_kv[0].expand(B, -1, -1, -1)
            sink_v = self.sink_kv[1].expand(B, -1, -1, -1)
            cmask = torch.cat([torch.ones(T, 1, dtype=torch.bool, device=dev), done], dim=1)
            rep = n_head // n_kv_head
            if "cmp" in BRANCHES:
                outs["cmp"] = self._attend(q, torch.cat([sink_k, ck], 2), torch.cat([sink_v, cv], 2), cmask)

            if "sel" in BRANCHES:
                # ---- 支路 2:挑选。打分直接复用 query 对摘要的注意力分数,不另算 ----
                with torch.no_grad():
                    score = (q @ ck.repeat_interleave(rep, dim=1).transpose(-1, -2)) / head_size ** 0.5
                    score = score.masked_fill(~done, float("-inf")).softmax(dim=-1).nan_to_num(0.0)
                    score = score.view(B, n_kv_head, rep, T, nb).sum(dim=2)   # 同组的头共用一次挑选
                    cur = CUR_BLOCK[:T].to(dev)
                    score = score.scatter(-1, cur.view(1, 1, T, 1).expand(B, n_kv_head, T, 1), 1e4)  # 当前块必选
                    score = score.masked_fill(~(done | F.one_hot(cur, nb).bool()), float("-inf"))
                    top = score.topk(min(top_n, nb), dim=-1)
                    chosen = torch.zeros(B, n_kv_head, T, nb, dtype=torch.bool, device=dev)
                    chosen.scatter_(-1, top.indices, top.values > float("-inf"))
                    tok_mask = chosen.repeat_interleave(block_len, dim=-1)[..., :T] & CAUSAL[:T, :T].to(dev)
                    self.last["chosen"] = chosen
                ks, vs = self._kv("sel", x)
                ks = apply_rope(ks, self.rope_cos, self.rope_sin)
                outs["sel"] = self._attend(q, ks, vs, tok_mask.repeat_interleave(rep, dim=1))

        if "win" in BRANCHES:
            # ---- 支路 3:滑窗 ----
            kw, vw = self._kv("win", x)
            outs["win"] = self._attend(q, apply_rope(kw, self.rope_cos, self.rope_sin), vw, WINDOW[:T, :T].to(dev))

        # ---- 门控加权:每个头自己决定三条支路各听几成 ----
        g = torch.sigmoid(self.gate(x)).view(B, T, 3, n_head).permute(0, 3, 1, 2)   # (B,nh,T,3)
        order = ["cmp", "sel", "win"]
        y = sum(g[..., order.index(b)].unsqueeze(-1) * outs[b] for b in BRANCHES)
        self.last["gate"] = {b: g[..., order.index(b)].mean().item() for b in BRANCHES}
        return self.drop(self.o_proj(y.transpose(1, 2).reshape(B, T, C)))

def tokens_read(t):
    """query 在位置 t(从 0 数)实际读取的 key 个数(去重之前的上界)。"""
    if FULL:
        return t + 1
    n = 0
    if "win" in BRANCHES:
        n += min(t + 1, window)
    if "sel" in BRANCHES:
        n += min(t + 1, top_n * block_len)
    if "cmp" in BRANCHES:
        n += (t + 1) // block_len + 1
    return n

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.ffn = SparseAttention(), SwiGLU()
        self.n1, self.n2 = RMSNorm(n_embd), RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.lm_head(self.norm_f(self.blocks(self.tok(idx))))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

model = Model().to(device)
n_param = sum(q.numel() for q in model.parameters())
seq_len = T_REC if args.task == "recall" else block_size   # 按实际训练序列长度统计,recall 样本只有 128 个位置
read_last = tokens_read(seq_len - 1)
read_avg = sum(tokens_read(t) for t in range(seq_len)) / seq_len
print(f"[{TAG}] 参数 {n_param/1e6:.3f} M | 最后一个 query 读 {read_last} 个 key(全注意力 {seq_len})"
      f" | 平均每个 query 读 {read_avg:.1f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def evaluate(iters):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        ls, hit, tot = [], 0, 0
        for _ in range(iters):
            x, y, _ = get_batch(split)
            logits, loss = model(x, y)
            ls.append(loss.item())
            if args.task == "recall":
                m = y != -100
                hit += (logits.argmax(-1)[m] == y[m]).sum().item()
                tot += m.sum().item()
        out[split] = sum(ls) / len(ls)
        if args.task == "recall":
            out[split + "_acc"] = hit / tot
    model.train()
    return out

curve = []
t0 = time.time()
print(f"===== 训练 [{TAG}/{args.task}] =====")
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        e = evaluate(eval_iters)
        row = [it, round(e["train"], 4), round(e["val"], 4)]
        if args.task == "recall":
            row.append(round(e["val_acc"], 4))
        curve.append(row)
        print(f"step {it:4d} | train {e['train']:.4f} | val {e['val']:.4f}"
              + (f" | 答对 {e['val_acc']*100:.1f}%" if args.task == "recall" else "") + f" | {time.time()-t0:.0f}s")
    x, y, _ = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

result = {"branches": TAG, "task": args.task, "params": n_param, "curve": curve,
          "block_len": block_len, "top_n": top_n, "window": window, "block_size": block_size,
          "tokens_read_last": read_last, "tokens_read_avg": round(read_avg, 1),
          "train_seconds": round(time.time() - t0)}

# ---- 给页面用的"挑中了哪几块":取一个 val 样本、最后一层 ----
if not FULL and "sel" in BRANCHES:
    model.eval()
    with torch.no_grad():
        x, y, where = get_batch("val")
        model(x[:1])
        att = model.blocks[-1].attn
        chosen = att.last["chosen"][0]                      # (nkv,T,nb)
        probe = list(range(block_len * 2 - 1, block_size, block_len * 2))
        if args.task == "recall":
            probe = list(range(Q0, T_REC, 2))
            result["key_positions"] = where[0]
        result["probe_positions"] = probe
        result["chosen_blocks"] = [[int(i) for i in chosen[0, t].nonzero().flatten().tolist()] for t in probe]
        if args.task == "lm":
            result["sample_text"] = "".join(itos[i] for i in x[0].tolist())
        result["gate_mean_last_layer"] = {k: round(v, 3) for k, v in att.last["gate"].items()}
        result["gate_mean_per_layer"] = []
        model(x)
        for b in model.blocks:
            result["gate_mean_per_layer"].append({k: round(v, 3) for k, v in b.attn.last["gate"].items()})

print(f"[{TAG}/{args.task}] 完成 | 用时 {result['train_seconds']}s")
if args.json:
    with open(args.json, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
