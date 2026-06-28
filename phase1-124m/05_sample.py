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
import time
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
ap.add_argument("--stream", action="store_true", help="流式输出(打字机效果);开启时强制 n=1")
ap.add_argument("--seed", type=int, default=1337, help="随机种子,固定它好对比")
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
    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # KV Cache：把这次新算的 k/v 接到历史 k/v 后面，省掉对老 token 的重复计算
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v)  # 存回去给下一步用
        # 因果掩码只在「一次喂多个新 token」(prompt 预填充)时需要；
        # 解码阶段每次只来 1 个新 query，它本就该看到全部历史，无需掩码
        is_causal = q.size(2) > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), present

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
    def forward(self, x, past_kv=None):
        attn_out, present = self.attn(self.ln_1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

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
    def forward(self, idx, past_kvs=None):
        B, T = idx.shape
        # 有 cache 时，新 token 的绝对位置要接着历史长度往后排
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        if past_kvs is None:
            past_kvs = [None] * len(self.transformer.h)
        presents = []
        for blk, past in zip(self.transformer.h, past_kvs):
            x, present = blk(x, past)
            presents.append(present)
        x = self.transformer.ln_f(x)
        return self.lm_head(x), presents

# ---- 加载 checkpoint ----
print(f"device={device}  加载 {args.ckpt} …")
ckpt = torch.load(args.ckpt, map_location=device)
cfg = GPTConfig(**ckpt["config"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"checkpoint 来自 step {ckpt.get('step','?')}, val_loss={ckpt.get('val_loss','?'):.4f}")

enc = tiktoken.get_encoding("gpt2")

def sync():
    """等 GPU 真正算完再读时钟，否则计时只量到「下发指令」"""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def sample_next(logits, temperature, top_k):
    """第②段：拿最后一个位置的全词表 logits → 裁剪 → 掷骰子，返回下一个 token"""
    logits = logits[:, -1, :]
    # 词表里 50257~50303 是为对齐 GPU 补的"空行"，从没训过，屏蔽掉防止解码报错
    logits[:, 50257:] = float("-inf")
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)

@torch.no_grad()
def generate_nocache(idx, max_new_tokens, temperature, top_k):
    """基线：每一步都把整段上下文重新喂进模型(O(T²) 重复计算)"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits, _ = model(idx_cond)
        idx = torch.cat([idx, sample_next(logits, temperature, top_k)], dim=1)
    return idx

class TokenStreamer:
    """流式打印：BPE 是字节级的，单个 token 可能只是半个 UTF-8 字符，
    凑不齐就先攒着，凑齐合法字符再打出来，避免中文/emoji 吐出乱码 �"""
    def __init__(self, enc):
        self.enc, self.pending = enc, []
    def add(self, tok_id):
        self.pending.append(tok_id)
        try:
            text = self.enc.decode_bytes(self.pending).decode("utf-8")
        except UnicodeDecodeError:
            return  # 半个字符，继续攒下一个 token
        print(text, end="", flush=True)
        self.pending = []

@torch.no_grad()
def generate_cached(idx, max_new_tokens, temperature, top_k, streamer=None):
    """KV Cache：prompt 只预填充一次，之后每步只喂「上一个新 token」+ 历史 KV"""
    logits, past = model(idx)                     # 预填充整个 prompt
    out = idx
    for step in range(max_new_tokens):
        nxt = sample_next(logits, temperature, top_k)
        out = torch.cat([out, nxt], dim=1)
        if streamer is not None:
            streamer.add(nxt[0].item())
        if step == max_new_tokens - 1:
            break
        logits, past = model(nxt, past)           # 只算这 1 个新 token
    return out

tokens = enc.encode(args.prompt)
print(f"\n===== prompt: {args.prompt!r} | temp={args.temperature} top_k={args.top_k} =====")

if args.stream:
    # 流式：强制单条，边算边吐字
    if args.n != 1:
        print(f"(流式模式强制 n=1，忽略 --n {args.n})")
    torch.manual_seed(args.seed)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    print(f"\n--- 样本 1 (stream) ---")
    print(args.prompt, end="", flush=True)
    streamer = TokenStreamer(enc)
    generate_cached(x, args.max_new_tokens, args.temperature, args.top_k, streamer=streamer)
    print()
else:
    # 非流式：跑 无Cache / 有Cache 两版，对比耗时 + 校验输出是否一致
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(args.n, 1)

    torch.manual_seed(args.seed); sync()
    t0 = time.time(); out_slow = generate_nocache(x, args.max_new_tokens, args.temperature, args.top_k); sync()
    t_slow = time.time() - t0

    torch.manual_seed(args.seed); sync()
    t0 = time.time(); out_fast = generate_cached(x, args.max_new_tokens, args.temperature, args.top_k); sync()
    t_fast = time.time() - t0

    n_tok = args.n * args.max_new_tokens
    same = torch.equal(out_slow, out_fast)
    print(f"\n⏱  无 Cache : {t_slow:6.2f}s  ({n_tok/t_slow:6.1f} tok/s)")
    print(f"⏱  有 Cache : {t_fast:6.2f}s  ({n_tok/t_fast:6.1f} tok/s)")
    print(f"🚀 提速      : {t_slow/t_fast:.2f}×")
    print(f"✅ 两版输出{'完全一致' if same else '有出入(数值误差导致采样分叉，正常)'}")

    for i in range(args.n):
        print(f"\n--- 样本 {i+1} ---")
        print(enc.decode(out_fast[i].tolist()))
