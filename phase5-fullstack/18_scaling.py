"""
里程碑 18(第 20 章):Scaling Law —— 用十几个小模型,预测 124M 训 10B token 的 loss
==============================================================================

第 6 章花了 4090 一整天,把 GPT-2 124M 训了 10B token,val loss 到 3.02。
能不能在训之前就算出这个数?Scaling law 的做法:

  【步骤 1】训一批小模型:参数量 N 从 0.6M 到 25M,每个配几档训练 token 数 D(5000 万到 16 亿)
  【步骤 2】每个 (N, D) 记最终 val loss,拟合  L(N, D) = E + A / N^α + B / D^β
            E 是这份数据本身的熵(模型再大、数据再多也降不下去的部分),
            A / N^α 是"模型太小"欠的账,B / D^β 是"数据太少"欠的账
  【步骤 3】把 N = 124M 的参数量、D = 10B 代进去,和第 6 章真实训出来的 loss 对照

这里的 N 按 Kaplan et al.(arXiv 2001.08361)的口径,只数"非嵌入参数"
(去掉 token 嵌入和位置嵌入);公式形式和拟合方法按 Hoffmann et al.(Chinchilla,
arXiv 2203.15556)§3.3:对 log L 做 Huber 回归。

所有小模型和第 6 章用同一份 GPT-2 结构(LayerNorm + GELU + 权重绑定)、同一个分词器、
同一份 FineWeb-Edu 数据、同样的 1024 上下文,只改层数 / 宽度 / 训练 token 数。

跑法:
    # 训一个点(需要 CUDA;数据同第 6 章,在 phase1-124m/data10b/)
    python 18_scaling.py train --n-layer 6 --n-embd 256 --n-head 4 --tokens 100e6 --json runs/s_L6_d256_D100M.json
    # 用同一套评测口径评第 6 章的两个 124M checkpoint
    python 18_scaling.py evalckpt --ckpt ../phase1-124m/ckpt10b/latest.pt --tokens 10e9 --json runs/s_124M_D10B.json
    # 拟合 + 外推(CPU 即可);L4_d192 · 4 亿、L6_d256 · 16 亿两个点训练发散(注意力 logit 增长),不参加拟合
    python 18_scaling.py fit runs/s_*.json --exclude L4_d192_D400M L6_d256_D1600M --out runs/ch20_fit.json
    # 排查发散:--diag 1 记注意力分数最大值、梯度范数与裁剪比例;--qk-norm 1 / --z-loss 1e-4 做对照
    python 18_scaling.py train --n-layer 4 --n-embd 192 --n-head 6 --tokens 400e6 --diag 1 --qk-norm 1 --json runs/diverged/diag2_qk.json
"""

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)

t = sub.add_parser("train", help="训一个 (N, D) 点")
t.add_argument("--n-layer", type=int, required=True)
t.add_argument("--n-embd", type=int, required=True)
t.add_argument("--n-head", type=int, required=True)
t.add_argument("--tokens", type=float, required=True, help="训练 token 数 D,如 100e6")
t.add_argument("--seed", type=int, default=1337, help="初始化与取数的随机种子;某个点训炸了换一个种子重跑")
t.add_argument("--batch-tokens", type=int, default=2**16, help="每步 token 数(65536)")
t.add_argument("--micro-batch", type=int, default=8, help="单次前向几条;只影响显存,梯度累积凑满 batch-tokens,结果不变")
t.add_argument("--lr", type=float, default=0.0, help="0 = 按宽度自动:min(3e-3, 6e-4 × 768 / n_embd)")
t.add_argument("--compile", type=int, default=1)
t.add_argument("--qk-norm", type=int, default=0, help="1 = 注意力里 q、k 先过 LayerNorm(qk-norm)")
t.add_argument("--z-loss", type=float, default=0.0, help="z-loss 系数:损失里加 z × mean(logsumexp(logits)²),压住输出 logit 整体漂移(0 = 不加)")
t.add_argument("--diag", type=int, default=0, help="1 = 每训 1/40 记一次诊断量(梯度范数、裁剪比例、嵌入表范数、logit 统计)")

e = sub.add_parser("evalckpt", help="用同一套评测口径评一个现成 checkpoint")
e.add_argument("--ckpt", type=str, required=True)
e.add_argument("--tokens", type=float, required=True, help="这个 checkpoint 训过多少 token(只记进 json)")

for sp in (t, e):
    sp.add_argument("--data-dir", type=str, default=os.path.join(HERE, "..", "phase1-124m", "data10b"))
    sp.add_argument("--eval-batches", type=int, default=40, help="val 评测:40 批 × 8 条 × 1024 = 327,680 token")
    sp.add_argument("--json", type=str, required=True)

f = sub.add_parser("fit", help="拟合 L(N, D) 并外推")
f.add_argument("files", nargs="+")
f.add_argument("--holdout", type=float, default=40e6, help="N 大于它的点不参加拟合,留作检验")
f.add_argument("--bootstrap", type=int, default=200, help="重抽拟合点再拟合的次数,用来给外推值一个区间")
f.add_argument("--max-d", type=float, default=0, help="D 大于它的点不参加拟合,留作检验(0 = 不限;留 1% 余量,4 亿档实际是 400,031,744)")
f.add_argument("--exclude", nargs="*", default=[], help="不参加拟合也不当检验的点,如 L4_d192_D400M(训练发散的点)")
f.add_argument("--out", type=str, required=True)
args = p.parse_args()

SEQ = 1024


# ---------------------------------------------------------------------------
# 模型:与 phase1-124m/04_gpt2_124m.py 完全一致,只是层数 / 宽度可调
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    qk_norm: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        hd = cfg.n_embd // cfg.n_head
        # qk-norm:q、k 各过一个 LayerNorm 再算分数,分数的量级就不会随训练一路变大
        self.q_ln, self.k_ln = (nn.LayerNorm(hd), nn.LayerNorm(hd)) if cfg.qk_norm else (None, None)
        self.probe = None   # 诊断时设成 list,前向会把本层注意力分数的最大值放进去

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if self.q_ln is not None:
            q, k = self.q_ln(q), self.k_ln(k)
        if self.probe is not None:
            att = (q @ k.transpose(-2, -1)).float() * (q.size(-1) ** -0.5)
            att = att.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
            self.probe.append(att.amax().item())
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02 * ((2 * self.cfg.n_layer) ** -0.5 if hasattr(m, "NANOGPT_SCALE_INIT") else 1)
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(T, device=idx.device))
        for blk in self.transformer.h:
            x = blk(x)
        logits = self.lm_head(self.transformer.ln_f(x)).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if self.training and getattr(self, "z_loss", 0.0) > 0:
            loss = loss + self.z_loss * logits.logsumexp(-1).pow(2).mean()
        return loss

    @torch.no_grad()
    def logit_stats(self, idx):
        """诊断用:一批输入上输出 logit 的整体水平(logsumexp 均值)、最大值,以及每层注意力分数的最大值"""
        T = idx.size(1)
        x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(T, device=idx.device))
        att_max = []
        for blk in self.transformer.h:
            blk.attn.probe = att_max
            x = blk(x)
            blk.attn.probe = None
        logits = self.lm_head(self.transformer.ln_f(x)).float()
        return logits.logsumexp(-1).mean().item(), logits.max().item(), logits.mean().item(), att_max

def count_params(model):
    """N = 非嵌入参数(Kaplan 口径)。lm_head 与 wte 共用一张表,只算一次,且算作嵌入。"""
    total = sum(p.numel() for p in model.parameters())
    emb = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    return total, total - emb


# ---------------------------------------------------------------------------
# 数据:和第 6 章同一份 .npy shard,按顺序读
# ---------------------------------------------------------------------------
class Shards:
    def __init__(self, split, B):
        self.files = sorted(glob.glob(os.path.join(args.data_dir, f"edufineweb_{split}_*.npy")))
        assert self.files, f"找不到 {split} shard:{args.data_dir}(先按第 6 章 prep_10b.sh 准备数据)"
        self.B, self.i = B, 0
        self._load(0)

    def _load(self, i):
        self.i, self.pos = i, 0
        self.tok = torch.from_numpy(np.load(self.files[i]).astype(np.int64))

    def next(self):
        n = self.B * SEQ + 1
        if self.pos + n > len(self.tok):
            self._load((self.i + 1) % len(self.files))
        chunk = self.tok[self.pos:self.pos + n]
        self.pos += self.B * SEQ
        return chunk[:-1].view(self.B, SEQ).cuda(non_blocking=True), chunk[1:].view(self.B, SEQ).cuda(non_blocking=True)

@torch.no_grad()
def val_loss(model):
    """固定评测:val shard 开头 eval_batches × 8 × 1024 个 token,每次都从头读,所有点口径相同"""
    model.eval()
    val = Shards("val", 8)
    tot = 0.0
    for _ in range(args.eval_batches):
        x, y = val.next()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += model(x, y).item()
    model.train()
    return tot / args.eval_batches


def train():
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    cfg = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, qk_norm=bool(args.qk_norm))
    model = GPT(cfg).cuda()
    model.z_loss = args.z_loss
    total, N = count_params(model)
    lr = args.lr or min(3e-3, 6e-4 * 768 / args.n_embd)
    steps = max(1, round(args.tokens / args.batch_tokens))
    D = steps * args.batch_tokens
    warm = min(100, max(1, steps // 20))
    accum = args.batch_tokens // (args.micro_batch * SEQ)
    assert accum * args.micro_batch * SEQ == args.batch_tokens
    print(f"L={args.n_layer} d={args.n_embd} h={args.n_head} | N(非嵌入)={N/1e6:.2f}M 总={total/1e6:.2f}M "
          f"| D={D/1e6:.0f}M token = {steps} 步 × {args.batch_tokens} | lr={lr:.2e} warmup={warm}", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    fwd = torch.compile(model) if args.compile else model

    def lr_at(s):   # warmup + 余弦退火到 10%,退火长度 = 这个点自己的总步数(Chinchilla §3.1 的要求)
        if s < warm:
            return lr * (s + 1) / warm
        r = (s - warm) / max(1, steps - warm)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * r)))

    data = Shards("train", args.micro_batch)
    curve, t0 = [], time.time()
    K = 40 if args.diag else 10
    marks = set(int(steps * k / K) for k in range(1, K))
    gn_sum, gn_clip, gn_n = 0.0, 0, 0
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        tot = 0.0
        for _ in range(accum):
            x, y = data.next()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = fwd(x, y) / accum
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item() if args.diag else \
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if args.diag:
            gn_sum, gn_clip, gn_n = gn_sum + gn, gn_clip + (gn > 1.0), gn_n + 1
            if s in marks:   # 裁剪前各参数的梯度范数(clip 是按比例整体缩,排序不变),取最大的 3 个
                scale = max(gn, 1.0)
                top = sorted(((p.grad.float().norm().item() * scale, n) for n, p in model.named_parameters()
                              if p.grad is not None), reverse=True)[:3]
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s in marks:
            v = val_loss(fwd)
            pt = {"tokens": (s + 1) * args.batch_tokens, "train": round(tot, 4), "val": round(v, 4)}
            if args.diag:
                lse, lmax, lmean, att_max = model.logit_stats(x[:4])
                pt.update(lr=lr_at(s), gnorm=round(gn_sum / gn_n, 4), clip_frac=round(gn_clip / gn_n, 3),
                          wte_rms=round(model.lm_head.weight.float().pow(2).mean().sqrt().item(), 5),
                          lnf_gain=round(model.transformer.ln_f.weight.float().abs().mean().item(), 4),
                          lse=round(lse, 3), logit_max=round(lmax, 3), logit_mean=round(lmean, 3),
                          att_max=[round(a, 2) for a in att_max], grad_top=[[n, round(g, 3)] for g, n in top])
                gn_sum, gn_clip, gn_n = 0.0, 0, 0
            curve.append(pt)
            el = time.time() - t0
            extra = (f" | gnorm {pt['gnorm']:.3f} clip {pt['clip_frac']:.2f} wte {pt['wte_rms']:.4f} "
                     f"lnf {pt['lnf_gain']:.3f} lse {pt['lse']:.2f} max {pt['logit_max']:.1f} att {pt['att_max']} "
                     f"top {[(n.replace('transformer.', ''), g) for n, g in pt['grad_top']]}") if args.diag else ""
            print(f"step {s+1}/{steps} | train {tot:.4f} | val {v:.4f}{extra} | {(s+1)*args.batch_tokens/el:,.0f} tok/s", flush=True)
    v = val_loss(fwd)
    el = time.time() - t0
    curve.append({"tokens": D, "train": round(tot, 4), "val": round(v, 4)})
    print(f"完成 | val {v:.4f} | {el/60:.1f} 分钟 | {D/el:,.0f} tok/s", flush=True)
    out = {"kind": "train", "n_layer": args.n_layer, "n_embd": args.n_embd, "n_head": args.n_head,
           "N": N, "N_total": total, "D": D, "C": 6 * N * D, "lr": lr, "seed": args.seed, "z_loss": args.z_loss, "qk_norm": args.qk_norm, "steps": steps,
           "val": round(v, 4), "curve": curve, "minutes": round(el / 60, 2), "tok_per_s": round(D / el)}
    json.dump(out, open(args.json, "w"), ensure_ascii=False, indent=1)


def evalckpt():
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg).cuda()
    model.load_state_dict(ck["model"])
    total, N = count_params(model)
    v = val_loss(model)
    print(f"{args.ckpt} | N(非嵌入)={N/1e6:.2f}M | D={args.tokens/1e9:.2f}B | val {v:.4f}(训练脚本记的 {ck.get('val_loss')})")
    json.dump({"kind": "ckpt", "ckpt": os.path.relpath(args.ckpt, HERE), "n_layer": cfg.n_layer, "n_embd": cfg.n_embd,
               "N": N, "N_total": total, "D": args.tokens, "C": 6 * N * args.tokens, "val": round(v, 4),
               "step": ck.get("step")}, open(args.json, "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# 拟合:L(N, D) = E + A / N^α + B / D^β
# 按 Chinchilla §3.3 / 附录 D.2:参数写成 a = log A, b = log B, e = log E,
#   log L̂ = LSE(a − α log N, b − β log D, e),对 log L 的残差用 Huber(δ = 1e-3),L-BFGS,多组初值取最好
# ---------------------------------------------------------------------------
def pred_log(th, n, d):
    """log L̂ = LSE(a − α log N, b − β log D, e);th 可以是 (5,) 或 (G, 5) 一批初值"""
    a, b, e, al, be = th.unbind(-1)
    ln, ld = n.log(), d.log()
    if th.dim() == 2:
        a, b, e, al, be = (x[:, None] for x in (a, b, e, al, be))
    return torch.logsumexp(torch.stack([a - al * ln, b - be * ld, e.expand_as(a - al * ln)]), 0)

def huber(r, delta=1e-3):
    return torch.where(r.abs() <= delta, 0.5 * r ** 2, delta * (r.abs() - 0.5 * delta))

# 初值网格同 Chinchilla 附录 D.2(6×6×5×5×5 = 4500 组)
GRID = torch.tensor([[a0, b0, e0, al0, be0]
                     for a0 in (0., 5., 10., 15., 20., 25.) for b0 in (0., 5., 10., 15., 20., 25.)
                     for e0 in (-1., -.5, 0., .5, 1.) for al0 in (0., .5, 1., 1.5, 2.)
                     for be0 in (0., .5, 1., 1.5, 2.)], dtype=torch.float64)

def fit_params(N, D, L, top=64):
    """先在 4500 组初值上算一遍损失,取最好的 top 组各跑 L-BFGS,返回损失最小的参数"""
    init = huber(pred_log(GRID, N, D) - L.log()).sum(-1)
    best = None
    for g in GRID[init.argsort()[:top]]:
        th = g.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([th], lr=1, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = huber(pred_log(th, N, D) - L.log()).sum()
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except RuntimeError:
            continue
        val = huber(pred_log(th, N, D) - L.log()).sum().item()
        if math.isfinite(val) and (best is None or val < best[0]):
            best = (val, th.detach().clone())
    return best[1]

def fit():
    pts = [json.load(open(fn)) for fn in args.files]
    tag_of = lambda q: f"L{q['n_layer']}_d{q['n_embd']}_D{round(q['D'] / 1e6)}M" if q["kind"] == "train" else q["ckpt"]

    def role(q):
        if tag_of(q) in args.exclude:
            return "excluded"
        if q["kind"] == "train" and q["N"] <= args.holdout and (not args.max_d or q["D"] <= args.max_d * 1.01):
            return "fit"
        return "check"

    miss = set(args.exclude) - {tag_of(q) for q in pts}
    assert not miss, f"--exclude 里的点不在输入文件中:{miss}"
    train_pts = [q for q in pts if role(q) == "fit"]
    N = torch.tensor([q["N"] for q in train_pts], dtype=torch.float64)
    D = torch.tensor([q["D"] for q in train_pts], dtype=torch.float64)
    L = torch.tensor([q["val"] for q in train_pts], dtype=torch.float64)
    print(f"拟合用 {len(train_pts)} 个点(N ≤ {args.holdout/1e6:.0f}M" + (f",D ≤ {args.max_d/1e6:.0f}M" if args.max_d else "")
          + (f";排除 {', '.join(args.exclude)}" if args.exclude else "") + ")")

    a, b, e, al, be = fit_params(N, D, L).tolist()
    A, B, E = math.exp(a), math.exp(b), math.exp(e)
    Lhat = lambda n, d: E + A / n ** al + B / d ** be
    print(f"L(N,D) = {E:.3f} + {A:.4g}/N^{al:.3f} + {B:.4g}/D^{be:.3f}")

    # 算力最优:固定 C = 6ND,解 dL/dN = 0 → N_opt = G (C/6)^(β/(α+β)),G = (αA / βB)^(1/(α+β))
    G = (al * A / (be * B)) ** (1 / (al + be))
    n_opt = lambda c: G * (c / 6) ** (be / (al + be))
    for c in (1e16, 1e17, 1e18, 1e19):
        n = n_opt(c)
        print(f"  C={c:.0e} → N_opt={n/1e6:.1f}M D_opt={c/6/n/1e6:.0f}M(D/N={c/6/n/n:.0f})")

    # bootstrap:有放回地重抽拟合点、重新拟合,看外推值能差多少(拟合点只有二十来个,单次拟合的数不能全信)
    held = [q for q in pts if role(q) == "check"]
    boot = {i: [] for i in range(len(held))}
    g = torch.Generator().manual_seed(0)
    for _ in range(args.bootstrap):
        idx = torch.randint(len(train_pts), (len(train_pts),), generator=g)
        th = fit_params(N[idx], D[idx], L[idx], top=8)
        for i, q in enumerate(held):
            v = pred_log(th, torch.tensor([float(q["N"])], dtype=torch.float64),
                         torch.tensor([float(q["D"])], dtype=torch.float64)).exp().item()
            boot[i].append(v)

    rows = []
    for q in pts:
        rows.append({**{k: q[k] for k in ("kind", "N", "D", "C", "val")},
                     "name": q.get("ckpt") or f"L{q['n_layer']}_d{q['n_embd']}",
                     "n_layer": q.get("n_layer"), "n_embd": q.get("n_embd"),
                     "pred": round(Lhat(q["N"], q["D"]), 4),
                     "in_fit": role(q) == "fit", "role": role(q),
                     "curve": q.get("curve")})
        if q in held and args.bootstrap:
            bs = sorted(boot[held.index(q)])
            rows[-1]["boot_lo"] = round(bs[int(0.05 * len(bs))], 4)
            rows[-1]["boot_hi"] = round(bs[int(0.95 * len(bs)) - 1], 4)
        tag = {"fit": "拟合", "check": "检验", "excluded": "排除"}[role(q)]
        extra = f" | bootstrap 90% [{rows[-1]['boot_lo']:.4f}, {rows[-1]['boot_hi']:.4f}]" if "boot_lo" in rows[-1] else ""
        print(f"  [{tag}] {rows[-1]['name']:<28} N={q['N']/1e6:7.2f}M D={q['D']/1e6:8.0f}M "
              f"实测 {q['val']:.4f} 预测 {rows[-1]['pred']:.4f} 差 {q['val']-rows[-1]['pred']:+.4f}{extra}")
    json.dump({"E": E, "A": A, "B": B, "alpha": al, "beta": be, "G": G, "holdout_N": args.holdout,
               "max_D": args.max_d, "exclude": args.exclude,
               "n_fit": len(train_pts), "bootstrap": args.bootstrap, "rows": rows},
              open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    {"train": train, "evalckpt": evalckpt, "fit": fit}[args.cmd]()
