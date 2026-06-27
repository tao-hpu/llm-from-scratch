"""
里程碑 5b：推理 / 采样 —— 玩一玩训出来的 base model
==================================================

加载 04 训练存下的 checkpoint，给一段开头(prompt)，让模型续写。

⚠️ 重要：这是【预训练基座模型】，只会"续写"，不会"回答问题"。
   给它 "The capital of China is"，它会接着编下去；
   给它 "Q: What is 2+2? A:"，它也只会续写文本，不会真答"4"。
   "会回答问题"是 Phase 2 的 SFT 教出来的，预训练阶段没有。

用法：
  python 05_sample.py --ckpt ckpt/latest.pt --prompt "The history of Rome" --n 3
  python 05_sample.py --ckpt ckpt/latest.pt --prompt "Once upon a time" --temperature 0.8 --top_k 40
"""
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, default="ckpt/latest.pt")
ap.add_argument("--prompt", type=str, default="The")
ap.add_argument("--max_new_tokens", type=int, default=200)
ap.add_argument("--n", type=int, default=3, help="生成几条样本")
ap.add_argument("--temperature", type=float, default=0.9, help="越高越随机，越低越保守")
ap.add_argument("--top_k", type=int, default=50, help="每步只从概率最高的 k 个里采样")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ---- 模型定义（与 04 完全一致，这样 state_dict 的键才对得上）----
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
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

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
        self.ln_1 = nn.LayerNorm(cfg.n_embd); self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd); self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

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
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for blk in self.transformer.h:
            x = blk(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

# ---- 加载 checkpoint ----
print(f"device={device}  加载 {args.ckpt} …")
ckpt = torch.load(args.ckpt, map_location=device)
cfg = GPTConfig(**ckpt["config"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"checkpoint 来自 step {ckpt.get('step','?')}, val_loss={ckpt.get('val_loss','?'):.4f}")

enc = tiktoken.get_encoding("gpt2")

@torch.no_grad()
def generate(idx, max_new_tokens, temperature, top_k):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits = model(idx_cond)[:, -1, :]
        # 词表里 50257~50303 是为对齐 GPU 补的"空行"，从没训过，屏蔽掉防止解码报错
        logits[:, 50257:] = float("-inf")
        logits = logits / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx

tokens = enc.encode(args.prompt)
x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(args.n, 1)
out = generate(x, args.max_new_tokens, args.temperature, args.top_k)

print(f"\n===== prompt: {args.prompt!r} | temp={args.temperature} top_k={args.top_k} =====")
for i in range(args.n):
    print(f"\n--- 样本 {i+1} ---")
    print(enc.decode(out[i].tolist()))
