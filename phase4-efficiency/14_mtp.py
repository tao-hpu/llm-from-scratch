"""
里程碑 14(第 16 章):投机解码与 MTP —— 先便宜地猜几个字,主模型一次 forward 验收
================================================================================

第 5 章的自回归生成,每吐一个 token 都要把主模型完整跑一遍。投机解码(speculative decoding)
的做法是:

  【步骤 1】draft   找个便宜的"草稿员"先猜 γ 个 token(小模型,或主模型自带的 MTP 头)
  【步骤 2】verify  主模型一次 forward 把这 γ 个位置全算出来,得到每个位置自己的概率 p
  【步骤 3】accept  逐个验收:草稿员给 x 的概率是 q(x),主模型给的是 p(x),
                    以 min(1, p(x)/q(x)) 的概率接受;一旦拒绝,就从 max(0, p−q) 归一化后的分布里重抽一个,
                    后面的草稿全部作废
  这套接受规则保证:最后吐出来的 token,分布和只用主模型一个一个采样**完全相同**(Leviathan et al. 2023)。
  省下来的是主模型 forward 的次数。

两种草稿员,都在这一关里真训一遍:

  draft-model  一个 1 层、64 维的小模型,自己一个一个猜 γ 个字
  MTP          主模型训练时顺带学"下下个字"(DeepSeek-V3 的 Multi-Token Prediction,深度 1):
               拿主模型最后一层的隐状态 h_t,拼上"下一个字"的嵌入,过一个小 Block,预测 t+2。
               推理时它就是现成的草稿员,一次猜 1 个。

跑法:
    python 14_mtp.py               # 训主模型 + MTP 头 + 小草稿模型,然后跑全部投机解码测量
    python 14_mtp.py --mtp 0 --no-spec   # 只训主模型、不带 MTP loss(看 MTP 对主模型 loss 的影响)
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 64
dropout = 0.1
lr = 1e-3
eval_interval = 500
eval_iters = 200

p = argparse.ArgumentParser()
p.add_argument("--mtp", type=int, default=1, help="1 = 训练时加 MTP 头(深度 1)")
p.add_argument("--lam", type=float, default=0.3, help="MTP loss 权重(DeepSeek-V3 前 10T token 用 0.3)")
p.add_argument("--max-iters", type=int, default=5000)
p.add_argument("--draft-iters", type=int, default=3000)
p.add_argument("--eval-iters", type=int, default=None)
p.add_argument("--no-spec", action="store_true", help="只训练,不做投机解码测量")
p.add_argument("--gen-tokens", type=int, default=600, help="每组测量生成多少 token")
p.add_argument("--json", type=str, default="")
args = p.parse_args()
if args.eval_iters is not None:
    eval_iters = args.eval_iters

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)

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
    return x.to(device), y.to(device)

# ===========================================================================
# 模型:第 12 章的 LLaMA 骨架,尺寸做成可配置(主模型和草稿模型共用一套代码)
# ===========================================================================
@dataclass
class Cfg:
    n_layer: int = 4
    n_embd: int = 128
    n_head: int = 4
    n_kv_head: int = 2

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
    def __init__(self, c):
        super().__init__()
        self.c, hs = c, c.n_embd // c.n_head
        self.q_proj = nn.Linear(c.n_embd, c.n_head * hs, bias=False)
        self.k_proj = nn.Linear(c.n_embd, c.n_kv_head * hs, bias=False)
        self.v_proj = nn.Linear(c.n_embd, c.n_kv_head * hs, bias=False)
        self.o_proj = nn.Linear(c.n_head * hs, c.n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        cos, sin = build_rope_cache(block_size, hs)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        c, hs = self.c, C // self.c.n_head
        q = self.q_proj(x).view(B, T, c.n_head, hs).transpose(1, 2)
        k = self.k_proj(x).view(B, T, c.n_kv_head, hs).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.n_kv_head, hs).transpose(1, 2)
        q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        rep = c.n_head // c.n_kv_head
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                                           is_causal=True, dropout_p=dropout if self.training else 0.0)
        return self.drop(self.o_proj(y.transpose(1, 2).reshape(B, T, C)))

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        hidden = (int(8 * c.n_embd / 3) + 7) // 8 * 8
        self.gate = nn.Linear(c.n_embd, hidden, bias=False)
        self.up = nn.Linear(c.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, c.n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn, self.ffn = Attention(c), SwiGLU(c)
        self.n1, self.n2 = RMSNorm(c.n_embd), RMSNorm(c.n_embd)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))

class GPT(nn.Module):
    def __init__(self, c, with_mtp=False):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(vocab_size, c.n_embd)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(c.n_layer)])
        self.norm_f = RMSNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, vocab_size)
        if with_mtp:
            # ---- MTP 模块(深度 1):[RMSNorm(h_t) ; RMSNorm(Emb(x_{t+1}))] → 线性合并 → 1 个 Block ----
            # 嵌入表和输出头都和主模型共用,自己只多一个投影和一个 Block
            self.mtp_norm_h = RMSNorm(c.n_embd)
            self.mtp_norm_e = RMSNorm(c.n_embd)
            self.mtp_proj = nn.Linear(2 * c.n_embd, c.n_embd, bias=False)
            self.mtp_block = Block(c)

    def forward(self, idx):
        """返回 (logits, 最后一层隐状态)。隐状态留给 MTP 用。"""
        h = self.blocks(self.tok(idx))
        return self.lm_head(self.norm_f(h)), h

    def mtp_logits(self, h, next_idx):
        """h: 位置 0..t 的主模型隐状态;next_idx: 位置 1..t+1 的 token。
        输出每个位置对"下下个 token"的预测(顺序模块,保持因果链)。"""
        z = torch.cat([self.mtp_norm_h(h), self.mtp_norm_e(self.tok(next_idx))], dim=-1)
        return self.lm_head(self.norm_f(self.mtp_block(self.mtp_proj(z))))

def train(model, iters, use_mtp, name):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    curve = []

    def losses(x, y):
        logits, h = model(x)
        main = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        if not use_mtp:
            return main, None
        # 位置 t 用 h_t 和 x_{t+1} 预测 x_{t+2}:x_{t+1} = x[:,1:],x_{t+2} = y[:,1:]
        ml = model.mtp_logits(h[:, :-1], x[:, 1:])
        return main, F.cross_entropy(ml.reshape(-1, vocab_size), y[:, 1:].reshape(-1))

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for split in ["train", "val"]:
            a, b = [], []
            for _ in range(eval_iters):
                m, t = losses(*get_batch(split))
                a.append(m.item())
                b.append(t.item() if t is not None else 0.0)
            out[split], out[split + "_mtp"] = sum(a) / len(a), sum(b) / len(b)
        model.train()
        return out

    t0 = time.time()
    for it in range(iters):
        if it % eval_interval == 0 or it == iters - 1:
            e = evaluate()
            curve.append([it, round(e["train"], 4), round(e["val"], 4), round(e["val_mtp"], 4)])
            print(f"[{name}] step {it:4d} | train {e['train']:.4f} | val {e['val']:.4f}"
                  + (f" | MTP 头 val {e['val_mtp']:.4f}" if use_mtp else "") + f" | {time.time()-t0:.0f}s")
        main, mtp = losses(*get_batch("train"))
        loss = main + (args.lam * mtp if use_mtp else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return curve

MAIN_CFG = Cfg()
DRAFT_CFG = Cfg(n_layer=1, n_embd=64, n_head=2, n_kv_head=1)

main_model = GPT(MAIN_CFG, with_mtp=bool(args.mtp)).to(device)
n_main = sum(q.numel() for n_, q in main_model.named_parameters() if not n_.startswith("mtp_"))
n_mtp = sum(q.numel() for n_, q in main_model.named_parameters() if n_.startswith("mtp_"))
print(f"主模型 {n_main/1e6:.3f} M | MTP 模块 {n_mtp/1e6:.3f} M")
main_curve = train(main_model, args.max_iters, bool(args.mtp), "main")
result = {"mtp": args.mtp, "lam": args.lam, "params_main": n_main, "params_mtp": n_mtp,
          "main_curve": main_curve, "main_val": main_curve[-1][2], "mtp_head_val": main_curve[-1][3]}

if args.no_spec:
    if args.json:
        json.dump(result, open(args.json, "w"), ensure_ascii=False, indent=1)
    raise SystemExit

draft_model = GPT(DRAFT_CFG).to(device)
n_draft = sum(q.numel() for q in draft_model.parameters())
print(f"草稿模型 {n_draft/1e6:.3f} M(主模型的 {100*n_draft/n_main:.0f}%)")
draft_curve = train(draft_model, args.draft_iters, False, "draft")
result.update({"params_draft": n_draft, "draft_curve": draft_curve, "draft_val": draft_curve[-1][2]})

# ===========================================================================
# 投机解码
# ===========================================================================
def sync():
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

def dist(logits, temp):
    """temp=0 表示贪心:返回 one-hot,接受规则自动退化成"草稿和主模型的 argmax 一致才接受"。"""
    if temp == 0:
        return F.one_hot(logits.argmax(-1), vocab_size).float()
    return F.softmax(logits / temp, dim=-1)

def sample(prob, gen):
    return int(torch.multinomial(prob.cpu(), 1, generator=gen).item())

def crop(seq):
    return torch.tensor([seq[-block_size:]], device=device)

class Counter:
    def __init__(self):
        self.main = self.draft = self.proposed = self.accepted = self.evaluated = 0
        self.min_pq = []          # 每次验收时 Σ min(p, q),它的期望就是论文里的 α

    def check(self, p, q, tok, gen):
        """一次验收。返回 (是否接受, 拒绝时重抽的 token)。"""
        self.evaluated += 1
        self.min_pq.append(torch.minimum(p, q).sum().item())
        ratio = (p[tok] / q[tok]).item() if q[tok] > 0 else 0.0
        if torch.rand(1, generator=gen).item() < min(1.0, ratio):
            self.accepted += 1
            return True, None
        resid = torch.clamp(p - q, min=0)
        return False, sample(resid / resid.sum(), gen)

@torch.no_grad()
def plain_generate(start, n_tok, temp, seed):
    gen = torch.Generator().manual_seed(seed)
    seq, c = list(start), Counter()
    while len(seq) - len(start) < n_tok:
        logits, _ = main_model(crop(seq))
        c.main += 1
        seq.append(sample(dist(logits[0, -1], temp), gen))
    return seq[len(start):], c

@torch.no_grad()
def spec_draft_model(start, n_tok, gamma, temp, seed):
    gen = torch.Generator().manual_seed(seed)
    seq, c = list(start), Counter()
    while len(seq) - len(start) < n_tok:
        # 【步骤 1】小模型自回归猜 gamma 个
        drafts, qs, ctx = [], [], list(seq)
        for _ in range(gamma):
            dl, _ = draft_model(crop(ctx))
            c.draft += 1
            q = dist(dl[0, -1], temp)
            tok = sample(q, gen)
            drafts.append(tok); qs.append(q); ctx.append(tok)
        c.proposed += gamma
        # 【步骤 2】主模型一次 forward,拿到 gamma+1 个位置的 p
        ml, _ = main_model(crop(seq + drafts))
        c.main += 1
        ps = dist(ml[0, -(gamma + 1):], temp)
        # 【步骤 3】逐个验收
        for i in range(gamma):
            ok, fix = c.check(ps[i], qs[i], drafts[i], gen)
            if ok:
                seq.append(drafts[i])
            else:
                seq.append(fix)
                break
        else:
            seq.append(sample(ps[gamma], gen))      # 全部接受:白送一个主模型自己的 token
    return seq[len(start):len(start) + n_tok], c

@torch.no_grad()
def spec_mtp(start, n_tok, temp, seed):
    """主模型自己的 MTP 头当草稿员。每次主模型 forward 吐 1 个或 2 个 token。"""
    gen = torch.Generator().manual_seed(seed)
    seq, c = list(start), Counter()
    draft = q_d = None
    while len(seq) - len(start) < n_tok:
        inp = seq + ([draft] if draft is not None else [])
        ml, h = main_model(crop(inp))
        c.main += 1
        L = min(len(inp), block_size)
        if draft is not None:
            c.proposed += 1
            ok, fix = c.check(dist(ml[0, -2], temp), q_d, draft, gen)
            if ok:
                seq.append(draft)
                seq.append(sample(dist(ml[0, -1], temp), gen))   # 草稿被接受,主模型顺手给出下一个
                h_t, cut = h[:, :L], 0
            else:
                seq.append(fix)
                h_t, cut = h[:, :L - 1], 1                        # 丢掉建立在错草稿上的最后一行
        else:
            seq.append(sample(dist(ml[0, -1], temp), gen))
            h_t, cut = h[:, :L], 0
        # 用 h_0..h_t 和 x_1..x_{t+1} 让 MTP 头猜 x_{t+2}
        nxt = crop(seq)[:, -h_t.size(1):]
        q_d = dist(main_model.mtp_logits(h_t, nxt)[0, -1], temp)
        c.draft += 1
        draft = sample(q_d, gen)
    return seq[len(start):len(start) + n_tok], c

def timed(fn, *a):
    sync(); t = time.time()
    out, c = fn(*a)
    sync()
    return out, c, time.time() - t

start = [stoi[ch] for ch in "ROMEO:\n"]
N = args.gen_tokens
spec = {"gen_tokens": N}

# ---- 一次 forward 验收多个位置的前提:因果掩码下,多喂几个 token 不改变前面位置的输出 ----
with torch.no_grad():
    ctx_c = val_data[1000:1000 + block_size].tolist()
    short = main_model(torch.tensor([ctx_c[:block_size - 8]], device=device))[0][0]
    full = main_model(torch.tensor([ctx_c], device=device))[0][0, :block_size - 8]
    spec["causal_logit_maxdiff"] = float((short - full).abs().max())
print(f"前 {block_size - 8} 个位置的 logits:单独算 vs 多喂 8 个 token 再算,最大差 {spec['causal_logit_maxdiff']:.2e}")

# ---- 正确性:贪心下,投机解码的输出必须和主模型自己贪心生成的逐字相同 ----
# 只在上下文窗口以内比:总长超过 block_size 以后,普通生成每步把窗口右移 1 格,
# 验收时却是把「已生成 + γ 个草稿」整体截成最后 block_size 个,靠后的位置看到的历史更短,
# 两边其实是两台窗口规则不同的生成器,逐字比较失去意义。真实系统用 KV cache,上下文远长于生成长度,没有这个问题。
PROMPT_LEN, GAMMAS = 8, [1, 4, 8]
GEN_LEN = block_size - PROMPT_LEN - max(GAMMAS)          # 最长一次输入 = 提示 + 已生成 + γ ≤ block_size
prompts = [val_data[i:i + PROMPT_LEN].tolist() for i in range(0, 20 * 997, 997)]
same = {f"draft_g{g}": 0 for g in GAMMAS}
same["mtp"] = 0
for pr in prompts:
    ref, _ = plain_generate(pr, GEN_LEN, 0, 0)
    for g in GAMMAS:
        same[f"draft_g{g}"] += spec_draft_model(pr, GEN_LEN, g, 0, 0)[0] == ref
    if args.mtp:
        same["mtp"] += spec_mtp(pr, GEN_LEN, 0, 0)[0] == ref
spec["greedy_check"] = {"prompts": len(prompts), "prompt_len": PROMPT_LEN, "tokens_each": GEN_LEN, "identical": same}
spec["greedy_identical_draft"] = all(same[f"draft_g{g}"] == len(prompts) for g in GAMMAS)
spec["greedy_identical_mtp"] = same["mtp"] == len(prompts) if args.mtp else None
print(f"贪心一致性({len(prompts)} 段提示 × {GEN_LEN} token,逐字相同的段数):{same}")

# ---- 正确性(采样):固定一个上下文,把"一步投机采样"重复 20 万次,看分布是否等于 p ----
with torch.no_grad():
    ctx = val_data[500:500 + block_size].tolist()
    p_ = F.softmax(main_model(crop(ctx))[0][0, -1], -1).cpu()
    q_ = F.softmax(draft_model(crop(ctx))[0][0, -1], -1).cpu()
    g = torch.Generator().manual_seed(7)
    M = 200_000
    xs = torch.multinomial(q_, M, replacement=True, generator=g)
    acc = torch.rand(M, generator=g) < torch.clamp(p_[xs] / q_[xs], max=1.0)
    resid = torch.clamp(p_ - q_, min=0); resid /= resid.sum()
    fixes = torch.multinomial(resid, M, replacement=True, generator=g)
    out = torch.where(acc, xs, fixes)
    emp = torch.bincount(out, minlength=vocab_size).float() / M
    tv = lambda a, b: 0.5 * (a - b).abs().sum().item()
    top = p_.topk(8).indices.tolist()
    spec["sampling_check"] = {"tv_spec_vs_p": round(tv(emp, p_), 4), "tv_q_vs_p": round(tv(q_, p_), 4),
                              "accept_rate": round(acc.float().mean().item(), 4),
                              "alpha_theory": round(torch.minimum(p_, q_).sum().item(), 4),
                              "context": "".join(itos[i] for i in ctx[-24:]),
                              "top_tokens": [itos[i] for i in top],
                              "p": [round(p_[i].item(), 4) for i in top],
                              "q": [round(q_[i].item(), 4) for i in top],
                              "spec": [round(emp[i].item(), 4) for i in top]}
    print(f"采样一致性:投机采样 vs p 的总变差距离 {tv(emp, p_):.4f}(草稿 q vs p:{tv(q_, p_):.4f})")

# ---- 单次 forward 耗时:草稿模型比主模型便宜多少 ----
with torch.no_grad():
    x = crop(ctx)
    for m in (main_model, draft_model):
        m(x)
    sync(); t = time.time()
    for _ in range(200):
        main_model(x)
    sync(); t_main = (time.time() - t) / 200
    t = time.time()
    for _ in range(200):
        draft_model(x)
    sync(); t_draft = (time.time() - t) / 200
spec["ms_per_forward"] = {"main": round(1000 * t_main, 3), "draft": round(1000 * t_draft, 3)}
print(f"单次 forward:主模型 {1000*t_main:.2f} ms | 草稿模型 {1000*t_draft:.2f} ms")

# ---- 主测量:temperature = 1.0 采样 ----
rows = []
_, c, sec = timed(plain_generate, start, N, 1.0, 1)
base_sec = sec
rows.append({"method": "plain", "gamma": 0, "main_forwards": c.main, "draft_forwards": 0,
             "tokens_per_main": round(N / c.main, 3), "alpha": None, "seconds": round(sec, 2)})
print(f"普通生成 {N} token:主模型 forward {c.main} 次,{sec:.1f}s")
for gamma in [1, 2, 3, 4, 6, 8]:
    _, c, sec = timed(spec_draft_model, start, N, gamma, 1.0, 1)
    alpha = c.accepted / max(c.evaluated, 1)
    rows.append({"method": "draft", "gamma": gamma, "main_forwards": c.main, "draft_forwards": c.draft,
                 "tokens_per_main": round(N / c.main, 3), "alpha": round(alpha, 4),
                 "alpha_minpq": round(sum(c.min_pq) / len(c.min_pq), 4),
                 "expected_tokens_per_main": round((1 - alpha ** (gamma + 1)) / (1 - alpha), 3) if alpha < 1 else gamma + 1,
                 "seconds": round(sec, 2), "speedup_wall": round(base_sec / sec, 3)})
    print(f"小草稿 γ={gamma}:接受率 {alpha:.3f} | 每次主 forward 吐 {N/c.main:.2f} 个 | {sec:.1f}s")
if args.mtp:
    _, c, sec = timed(spec_mtp, start, N, 1.0, 1)
    alpha = c.accepted / max(c.evaluated, 1)
    rows.append({"method": "mtp", "gamma": 1, "main_forwards": c.main, "draft_forwards": c.draft,
                 "tokens_per_main": round(N / c.main, 3), "alpha": round(alpha, 4),
                 "alpha_minpq": round(sum(c.min_pq) / len(c.min_pq), 4),
                 "expected_tokens_per_main": round(1 + alpha, 3),
                 "seconds": round(sec, 2), "speedup_wall": round(base_sec / sec, 3)})
    print(f"MTP 自草稿:接受率 {alpha:.3f} | 每次主 forward 吐 {N/c.main:.2f} 个 | {sec:.1f}s")
spec["rows"] = rows
result["spec"] = spec

# ---- 一段真实样本,标出哪些字是草稿被接受的 ----
@torch.no_grad()
def traced_mtp(n_tok, temp, seed):
    gen = torch.Generator().manual_seed(seed)
    seq, marks = list(start), []
    draft = q_d = None
    c = Counter()
    while len(seq) - len(start) < n_tok:
        inp = seq + ([draft] if draft is not None else [])
        ml, h = main_model(crop(inp))
        L = min(len(inp), block_size)
        if draft is not None:
            ok, fix = c.check(dist(ml[0, -2], temp), q_d, draft, gen)
            if ok:
                seq += [draft, sample(dist(ml[0, -1], temp), gen)]; marks += ["a", "m"]
                h_t = h[:, :L]
            else:
                seq.append(fix); marks.append("r")
                h_t = h[:, :L - 1]
        else:
            seq.append(sample(dist(ml[0, -1], temp), gen)); marks.append("m")
            h_t = h[:, :L]
        nxt = crop(seq)[:, -h_t.size(1):]
        q_d = dist(main_model.mtp_logits(h_t, nxt)[0, -1], temp)
        draft = sample(q_d, gen)
    return "".join(itos[i] for i in seq[len(start):len(start) + n_tok]), "".join(marks)[:n_tok]

if args.mtp:
    txt, marks = traced_mtp(240, 1.0, 3)
    result["mtp_trace"] = {"text": txt, "marks": marks}
    print(txt)

if args.json:
    with open(args.json, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
