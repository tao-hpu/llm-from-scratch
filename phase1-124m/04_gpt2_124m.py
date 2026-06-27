"""
里程碑 5：复现 GPT-2 124M —— 从"玩具"到"真规模"
=================================================

这是 03 的直接升级。架构内核（注意力+FFN+残差+LayerNorm）一模一样，
但加上了 build-nanogpt 相比我们玩具版多出来的全部"工程机制"。
每一样我都在代码里标了【新】，并说明它解决什么问题：

  架构层（更大 + 几个细节）
  【新-1】真规模：12层 / 12头 / 768维 / 上下文1024 / 词表50257（GPT-2 124M 原配）
  【新-2】Flash-Attention：用 F.scaled_dot_product_attention，省显存、更快
  【新-3】权重绑定(weight tying)：输入嵌入和输出投影共享一张表（省参数、是 GPT-2 原设计）
  【新-4】残差缩放初始化：深层残差累加会让方差爆炸，按 1/sqrt(2*层数) 缩小初始化

  训练层（让大模型训得稳、训得动）
  【新-5】bf16 混合精度：4090 上算得更快、省一半显存，精度几乎无损
  【新-6】梯度累积：显存装不下大 batch，就拆成小份累加梯度，等效大 batch
  【新-7】warmup + 余弦学习率：先慢慢热身再衰减，大模型不热身开局就崩
  【新-8】梯度裁剪：把梯度范数截到 1.0，防止偶发的巨大梯度炸掉训练
  【新-9】权重衰减分组 + fused AdamW：只对矩阵类参数做 weight decay，优化器融合加速
  【新-10】TF32：矩阵乘用 TF32，4090 上免费提速

数据来自 prepare_fineweb.py 产出的 .npy shards（真 BPE 分词的 FineWeb-Edu）。

用法：python 04_gpt2_124m.py --max_steps 2000
"""
import os
import glob
import math
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--data_dir", type=str, default="data")
p.add_argument("--total_batch_size", type=int, default=524288, help="每步等效 token 数 (2**19)")
p.add_argument("--micro_batch", type=int, default=8, help="单次前向的序列数 B（显存不够就调小）")
p.add_argument("--seq_len", type=int, default=1024, help="上下文长度 T")
p.add_argument("--max_steps", type=int, default=2000)
p.add_argument("--warmup_steps", type=int, default=100)
p.add_argument("--max_lr", type=float, default=6e-4)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--compile", type=int, default=1, help="是否 torch.compile（1 开 0 关）")
p.add_argument("--out_dir", type=str, default="ckpt")
args = p.parse_args()
min_lr = args.max_lr * 0.1

device = "cuda"
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.set_float32_matmul_precision("high")   # 【新-10】TF32：矩阵乘用 TF32 精度，免费提速

# ---------------------------------------------------------------------------
# 模型：GPT-2 124M
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304   # 50257 向上取整到 128 的倍数，GPU 上更高效（多出的行永远学不到，无害）
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # 一次性算出 Q,K,V
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1                    # 标记：这层要做残差缩放初始化
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 拆成多头：(B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 【新-2】Flash-Attention：is_causal=True 自动做因果掩码，且不显式构造 T×T 矩阵 -> 省显存
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU(approximate="tanh")   # GPT-2 用的近似 GELU
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    """和 03 一样的 pre-norm 残差结构，只是子层换成了 GPT-2 标准实现。"""
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 沟通
        x = x + self.mlp(self.ln_2(x))    # 思考
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token 嵌入
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # 位置嵌入
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 【新-3】权重绑定：输入嵌入表 == 输出投影矩阵。少 ~40M 参数，且是 GPT-2 原设计。
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # 【新-4】残差缩放初始化：残差路径上每加一个子层，方差就累加一次。
            # 把这些投影层的初始化标准差按 1/sqrt(2*n_layer) 缩小，抵消累加，开局更稳。
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.cfg.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, lr):
        # 【新-9】权重衰减分组：只对 2 维参数（矩阵、嵌入）做 weight decay，
        # 1 维参数（bias、LayerNorm 的 gain/bias）不做 —— 这是被验证更好的惯例。
        params = {n: pp for n, pp in self.named_parameters() if pp.requires_grad}
        decay = [pp for pp in params.values() if pp.dim() >= 2]
        nodecay = [pp for pp in params.values() if pp.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        ndecay = sum(pp.numel() for pp in decay)
        nnodecay = sum(pp.numel() for pp in nodecay)
        print(f"  decay 参数: {len(decay)} 张, {ndecay:,} 个 | no-decay: {len(nodecay)} 张, {nnodecay:,} 个")
        # fused=True：把优化器更新融合成一个 CUDA kernel，更快
        opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return opt

# ---------------------------------------------------------------------------
# 数据加载：读 .npy shards
# ---------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B, self.T = B, T
        pattern = os.path.join(args.data_dir, f"edufineweb_{split}_*.npy")
        self.shards = sorted(glob.glob(pattern))
        assert self.shards, f"找不到 {split} 数据：{pattern}"
        print(f"  {split}: {len(self.shards)} 片 shard")
        self.reset()

    def _load(self, path):
        arr = np.load(path).astype(np.int32)
        return torch.tensor(arr, dtype=torch.long)

    def reset(self):
        self.cur_shard = 0
        self.tokens = self._load(self.shards[0])
        self.pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        chunk = self.tokens[self.pos: self.pos + B * T + 1]
        # 如果当前 shard 剩余不够一个 batch，换下一片（循环）
        if len(chunk) < B * T + 1:
            self.cur_shard = (self.cur_shard + 1) % len(self.shards)
            self.tokens = self._load(self.shards[self.cur_shard])
            self.pos = 0
            chunk = self.tokens[self.pos: self.pos + B * T + 1]
        x = chunk[:-1].view(B, T)
        y = chunk[1:].view(B, T)
        self.pos += B * T
        return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# 学习率调度：【新-7】warmup + 余弦衰减
# ---------------------------------------------------------------------------
def get_lr(step):
    if step < args.warmup_steps:                       # 线性热身
        return args.max_lr * (step + 1) / args.warmup_steps
    if step >= args.max_steps:                         # 训练末尾保持最小
        return min_lr
    # 中间：余弦从 max_lr 平滑衰减到 min_lr
    ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (args.max_lr - min_lr)

# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def main():
    os.makedirs(args.out_dir, exist_ok=True)
    B, T = args.micro_batch, args.seq_len
    assert args.total_batch_size % (B * T) == 0
    # 【新-6】梯度累积步数：等效大 batch / 单次微 batch
    grad_accum = args.total_batch_size // (B * T)
    print(f"等效 batch = {args.total_batch_size:,} tokens | 微 batch B={B}, T={T} | 梯度累积 {grad_accum} 步")

    train_loader = DataLoaderLite(B, T, "train")
    val_loader = DataLoaderLite(B, T, "val")

    model = GPT(GPTConfig()).to(device)
    nparams = sum(pp.numel() for pp in model.parameters())
    print(f"参数量 = {nparams/1e6:.1f} M")
    if args.compile:
        print("torch.compile 编译中（首次约 1 分钟）…")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(weight_decay=0.1, lr=args.max_lr)

    logf = open(os.path.join(args.out_dir, "train.log"), "a")
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"# 开始训练 | ln(50257)={math.log(50257):.4f}（初始 loss 该在这附近）")

    for step in range(args.max_steps):
        t0 = time.time()
        last = (step == args.max_steps - 1)

        # ---- 周期性验证 ----
        if step % args.eval_every == 0 or last:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                vloss = 0.0
                vsteps = 20
                for _ in range(vsteps):
                    xv, yv = val_loader.next_batch()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, l = model(xv, yv)
                    vloss += l.item() / vsteps
            log(f"step {step:5d} | val loss {vloss:.4f}")
            # 存 checkpoint
            ckpt = {"model": (model._orig_mod if args.compile else model).state_dict(),
                    "step": step, "val_loss": vloss, "config": GPTConfig().__dict__}
            torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))
            model.train()

        # ---- 一个优化步 = grad_accum 个微 batch 累积 ----
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(grad_accum):
            x, y = train_loader.next_batch()
            # 【新-5】bf16 混合精度：前向在 bf16 下算
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum     # 累积要除以步数，等效于对大 batch 求平均
            loss_accum += loss.item()
            loss.backward()
        # 【新-8】梯度裁剪：范数截到 1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 设这一步的学习率
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()

        dt = time.time() - t0
        tok_per_sec = args.total_batch_size / dt
        if step % 10 == 0 or last:
            mem = torch.cuda.max_memory_allocated() / 1e9
            log(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | norm {norm:.2f} "
                f"| {dt*1000:.0f}ms | {tok_per_sec:,.0f} tok/s | mem {mem:.1f}GB")

    log("# 训练结束")
    logf.close()

if __name__ == "__main__":
    main()
