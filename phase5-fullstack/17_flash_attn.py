"""
里程碑 17(第 19 章):手写 FlashAttention —— 把 F.scaled_dot_product_attention 这个黑盒拆开
==============================================================================

从第 6 章起,注意力一直是一行 F.scaled_dot_product_attention(q, k, v, is_causal=True)。
它和第 2 章手写的注意力算的是同一个数,区别在"中间结果放在哪":

  朴素写法  S = QKᵀ/√d → softmax → P·V。S 和 P 都是 T×T,整块写回显存再读出来。
            T = 16384、12 个头、batch 4 时光 S 就要 4×12×16384²×4 字节 ≈ 51 GB
  Flash     把 Q、K、V 切成小块,一次只把一块 Q 和一块 K/V 读进片上 SRAM,
            块内算完直接累加到输出里,T×T 的 S 和 P 从不写回显存

难点在 softmax:它要先知道一整行的最大值和总和,而分块时一行被切成了好几段。解法是
【online softmax】每读一块,记下"到目前为止的最大值 m 和指数和 l";新的一块带来更大的最大值时,
               把之前累加的结果乘上 exp(m_旧 − m_新) 缩回去。读完最后一块,输出 ÷ l 就是精确的 softmax·V
这不是近似,结果和朴素写法只差浮点舍入。

本脚本三层实现,逐层对照:
  【1】naive_attention       第 2 章的写法,T×T 全部落地
  【2】blocked_attention     纯 PyTorch 的分块 + online softmax(算法和 FlashAttention-2 前向一致,Python 循环很慢,只用来对数)
  【3】flash_attn_triton     同一个算法写成 Triton kernel,一个 program 负责一块 Q,块内的矩阵乘和 softmax 都在片上做
然后:① 三者和 SDPA 对数;② 不同 T 下测时间和峰值显存;③ 把第 6 章 124M 的注意力换成【3】,val loss 应当不变。

出处:Dao et al., FlashAttention(arXiv 2205.14135)Algorithm 1;FlashAttention-2(arXiv 2307.08691)Algorithm 1
     把 1/l 的除法挪到最后、按 Q 块并行。online softmax 的递推见 Milakov & Gimelshein(arXiv 1805.02867)。
这里只写了前向(推理够用);训练还要反向 kernel,FlashAttention 论文 §3.1 的做法是反向时按块重算 S,不存它。

跑法:
    python 17_flash_attn.py --json runs/ch19_flash.json          # CUDA:全部
    # MPS / CPU:没有 Triton;【1】【2】与 SDPA 对数照跑,计时只测 T ≤ 2048、不测显存
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
p.add_argument("--ckpt", type=str, default=os.path.join(HERE, "..", "phase1-124m", "ckpt10b", "latest.pt"))
p.add_argument("--data", type=str, default=None, help="默认依次找 phase1-124m/data10b、phase1-124m/data、phase4-efficiency/data 下的 val shard")
p.add_argument("--lengths", type=str, default="", help="逗号分隔;默认 CUDA 测 512…16384,MPS / CPU 测 256…2048")
p.add_argument("--batch", type=int, default=4)
p.add_argument("--heads", type=int, default=12)
p.add_argument("--head-dim", type=int, default=64)
p.add_argument("--json", type=str, default="")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
HAS_TRITON = False
if device == "cuda":
    try:
        import triton
        import triton.language as tl
        HAS_TRITON = True
    except ImportError:
        print("没装 triton,跳过【3】(Linux 上装 CUDA 版 PyTorch 会自带 triton)")


# ---------------------------------------------------------------------------
# 【1】朴素注意力:T×T 的分数矩阵整块落地
# ---------------------------------------------------------------------------
def naive_attention(q, k, v):
    T, d = q.size(-2), q.size(-1)
    s = (q @ k.transpose(-2, -1)) / math.sqrt(d)                      # (B, H, T, T)
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return F.softmax(s.float(), dim=-1).to(q.dtype) @ v


# ---------------------------------------------------------------------------
# online softmax:一行数分几段读,只记 m(最大值)和 l(指数和)
# ---------------------------------------------------------------------------
def online_softmax_trace(x, block):
    """返回每读完一段后的 (m, l),最后 exp(x − m) / l 就是 softmax(x)。给页面画图用"""
    m, l, trace = float("-inf"), 0.0, []
    for i in range(0, len(x), block):
        seg = x[i:i + block]
        m_new = max(m, max(seg))
        l = l * math.exp(m - m_new) + sum(math.exp(v - m_new) for v in seg)   # 旧的和先按新最大值缩放
        m = m_new
        trace.append({"read": i + len(seg), "m": m, "l": l})
    return trace


# ---------------------------------------------------------------------------
# 【2】纯 PyTorch 分块:算法与 FlashAttention-2 前向一致
# ---------------------------------------------------------------------------
def blocked_attention(q, k, v, Br=64, Bc=64):
    B, H, T, d = q.shape
    scale = 1 / math.sqrt(d)
    out = torch.empty_like(q)
    for i in range(0, T, Br):                          # 外循环:一块 Q(对应 Triton 里的一个 program)
        qi = q[:, :, i:i + Br].float()
        rows = torch.arange(i, min(i + Br, T), device=q.device)
        m = torch.full((B, H, len(rows)), float("-inf"), device=q.device)
        l = torch.zeros(B, H, len(rows), device=q.device)
        acc = torch.zeros(B, H, len(rows), d, device=q.device)
        for j in range(0, min(i + Br, T), Bc):         # 内循环:只走到对角线,右上方的块因果掩码全遮,直接跳过
            kj, vj = k[:, :, j:j + Bc].float(), v[:, :, j:j + Bc].float()
            s = (qi @ kj.transpose(-2, -1)) * scale    # (B, H, Br, Bc):只有这一小块在"片上"
            cols = torch.arange(j, min(j + Bc, T), device=q.device)
            s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
            m_new = torch.maximum(m, s.amax(-1))
            p_ = torch.exp(s - m_new[..., None])
            alpha = torch.exp(m - m_new)               # 最大值变大了,之前累加的要缩回去
            l = l * alpha + p_.sum(-1)
            acc = acc * alpha[..., None] + p_ @ vj
            m = m_new
        out[:, :, i:i + Br] = (acc / l[..., None]).to(q.dtype)   # 除以 l 只在最后做一次(FlashAttention-2 的改动)
    return out


# ---------------------------------------------------------------------------
# 【3】Triton kernel:同一个算法,一个 program 负责一块 BLOCK_M 行的 Q
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _flash_fwd(Q, K, V, O, sm_scale,
                   s_b, s_h, s_t,                      # q/k/v/o 都是连续的 (B, H, T, d),共用步长
                   H, T,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
        pid_m = tl.program_id(0)                       # 第几块 Q
        pid_bh = tl.program_id(1)                      # 第几个 (batch, head)
        base = (pid_bh // H) * s_b + (pid_bh % H) * s_h
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, D)
        q = tl.load(Q + base + rows[:, None] * s_t + dims[None, :], mask=rows[:, None] < T, other=0.0)

        m = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        scale = sm_scale * 1.4426950408889634          # exp(x) = exp2(x · log2 e),exp2 在 GPU 上更快
        hi = tl.minimum((pid_m + 1) * BLOCK_M, T)      # 因果:只读到对角线
        for start in range(0, hi, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            k = tl.load(K + base + cols[None, :] * s_t + dims[:, None], mask=cols[None, :] < T, other=0.0)  # (D, BLOCK_N)
            s = tl.dot(q, k) * scale                   # (BLOCK_M, BLOCK_N),留在片上
            s = tl.where((rows[:, None] >= cols[None, :]) & (cols[None, :] < T), s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, 1))
            p_ = tl.math.exp2(s - m_new[:, None])
            alpha = tl.math.exp2(m - m_new)
            l = l * alpha + tl.sum(p_, 1)
            v = tl.load(V + base + cols[:, None] * s_t + dims[None, :], mask=cols[:, None] < T, other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p_.to(v.dtype), v)
            m = m_new
        acc = acc / l[:, None]
        tl.store(O + base + rows[:, None] * s_t + dims[None, :], acc.to(O.dtype.element_ty), mask=rows[:, None] < T)

    def flash_attn_triton(q, k, v, BLOCK_M=64, BLOCK_N=64):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        B, H, T, d = q.shape
        o = torch.empty_like(q)
        grid = (triton.cdiv(T, BLOCK_M), B * H)
        _flash_fwd[grid](q, k, v, o, 1 / math.sqrt(d), q.stride(0), q.stride(1), q.stride(2), H, T,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=d, num_warps=4, num_stages=2)
        return o


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# ---------------------------------------------------------------------------
# 计时 / 显存
# ---------------------------------------------------------------------------
def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def measure(fn, q, k, v, reps=10):
    """返回 (毫秒, 注意力本身额外占的峰值显存 MB);显存爆了返回 None"""
    try:
        fn(q, k, v); sync()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        ts = []
        for _ in range(reps):
            sync(); t0 = time.perf_counter()
            fn(q, k, v)
            sync(); ts.append(time.perf_counter() - t0)
        mem = (torch.cuda.max_memory_allocated() - base) / 2**20 if device == "cuda" else None
        return round(sorted(ts)[len(ts) // 2] * 1000, 3), (round(mem, 1) if mem is not None else None)
    except RuntimeError as e:   # CUDA / MPS / CPU 的显存(内存)不够都是 RuntimeError 的子类或本身,按报错文字认
        if not any(k in str(e).lower() for k in ("out of memory", "can't allocate", "cannot allocate")):
            raise
        if device == "cuda":
            torch.cuda.empty_cache()
        return None


# ---------------------------------------------------------------------------
# 第 6 章的 124M:注意力换成可替换的函数
# ---------------------------------------------------------------------------
ATTN = sdpa

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
        y = ATTN(q, k, v)
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
        x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(idx.size(1), device=idx.device))
        for blk in self.transformer.h:
            x = blk(x)
        return self.lm_head(self.transformer.ln_f(x))

def find_val():
    cands = [args.data] if args.data else [os.path.join(HERE, "..", d, "edufineweb_val_000000.npy") for d in
                                           ("phase1-124m/data10b", "phase1-124m/data", "phase4-efficiency/data")]
    return next((c for c in cands if c and os.path.exists(c)), None)


def main():
    global ATTN
    out = {"device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
           "triton": HAS_TRITON, "config": vars(args)}
    dt = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} | dtype={dt} | triton={HAS_TRITON}")

    # ① online softmax 小例子
    x = [1.0, 3.0, 0.5, 2.0, 6.0, 1.5, 4.0, 0.0]
    tr = online_softmax_trace(x, 2)
    exact = [math.exp(v - max(x)) / sum(math.exp(u - max(x)) for u in x) for v in x]
    online = [math.exp(v - tr[-1]["m"]) / tr[-1]["l"] for v in x]
    out["online_softmax"] = {"x": x, "block": 2, "trace": tr, "max_err": max(abs(a - b) for a, b in zip(exact, online))}
    print(f"online softmax:8 个数分 4 段读,与一次算完的最大误差 {out['online_softmax']['max_err']:.2e}")

    # ② 对数:同一组 q/k/v,各实现与 SDPA 的最大绝对误差
    torch.manual_seed(0)
    B, H, d = 2, args.heads, args.head_dim
    check = {}
    for T in (256, 1000):                              # 1000 不是块大小的整数倍,检查边界处理
        q, k, v = (torch.randn(B, H, T, d, device=device, dtype=dt) for _ in range(3))
        ref = sdpa(q.float(), k.float(), v.float())
        impls = {"naive": naive_attention, "blocked": blocked_attention}
        if HAS_TRITON:
            impls["triton"] = flash_attn_triton
        impls["sdpa"] = sdpa
        check[T] = {n: float((fn(q, k, v).float() - ref).abs().max()) for n, fn in impls.items()}
        print(f"  T={T:5d} 与 fp32 SDPA 的最大绝对误差:" + " | ".join(f"{n} {e:.2e}" for n, e in check[T].items()))
    out["check"] = check

    # ③ 时间 / 显存
    bench = []
    impls = [("naive", naive_attention), ("sdpa", sdpa)] + ([("triton", flash_attn_triton)] if HAS_TRITON else [])
    lengths = args.lengths or ("512,1024,2048,4096,8192,16384" if device == "cuda" else "256,512,1024,2048")
    for T in [int(s) for s in lengths.split(",")]:
        q, k, v = (torch.randn(args.batch, args.heads, T, args.head_dim, device=device, dtype=dt) for _ in range(3))
        row = {"T": T, "S_MB": round(args.batch * args.heads * T * T * q.element_size() / 2**20, 1),
               "qkvo_MB": round(4 * q.numel() * q.element_size() / 2**20, 1)}
        for name, fn in impls:
            r = measure(fn, q, k, v)
            row[name] = {"ms": r[0], "mem_MB": r[1]} if r else None
        # 注意力的计算量:因果只算一半,QKᵀ 与 PV 各 2·T²·d 次乘加 → 约 2·B·H·T²·d FLOPs
        flops = 2 * args.batch * args.heads * T * T * args.head_dim
        for name, _ in impls:
            if row[name]:
                row[name]["tflops"] = round(flops / (row[name]["ms"] / 1000) / 1e12, 1)
        bench.append(row)
        print(f"  T={T:6d} | " + " | ".join(
            f"{n} " + ((f"{row[n]['ms']:8.2f} ms " + (f"{row[n]['mem_MB']:8.0f} MB" if row[n]['mem_MB'] is not None else "显存未测(仅 CUDA)"))
                       if row[n] else "   显存不够   ") for n, _ in impls))
        del q, k, v
    out["bench"] = bench

    # ④ 换进 124M:同一批 val token,SDPA 与 Triton 各算一次 loss
    path = find_val()
    if os.path.exists(args.ckpt) and path:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        model = GPT(GPTConfig(**ck["config"]))
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        tok = torch.from_numpy(np.load(path, mmap_mode="r")[:8 * 1024 * 8 + 1].astype(np.int64))
        losses = {}
        for name, fn in [("sdpa", sdpa), ("naive", naive_attention)] + ([("triton", flash_attn_triton)] if HAS_TRITON else []):
            ATTN = fn
            tot = 0.0
            with torch.no_grad():
                for i in range(8):
                    c = tok[i * 8192:(i + 1) * 8192 + 1].to(device)
                    x, y = c[:-1].view(8, 1024), c[1:].view(8, 1024)
                    with torch.autocast(device_type=device, dtype=dt, enabled=(device == "cuda")):
                        logits = model(x)
                    tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).item() / 8
            losses[name] = round(tot, 5)
        ATTN = sdpa
        out["model_loss"] = {"tokens": 8 * 8 * 1024, "loss": losses}
        print(f"124M 在 65,536 个 val token 上的 loss:" + " | ".join(f"{n} {v:.5f}" for n, v in losses.items()))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(out, open(args.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
