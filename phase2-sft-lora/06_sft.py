"""
里程碑 6：SFT（监督微调）—— 把"只会续写的 base"调成"会听指令答题"
========================================================================

这是 Phase 2 的第一步。它**直接复用 Phase 1 训出来的 124M base 模型**
（phase1-124m/ckpt10b/latest.pt），在上面做一遍监督微调。

核心结论先记住：**SFT 和预训练用的是同一个 loss（预测下一个 token 的交叉熵），
没有新数学。** 改动只有三处，代码里都标了【SFT】：

  【SFT-1】数据：从"一坨连续文本"换成「指令 → 回答」成对数据。
  【SFT-2】对话模板 + EOS：把指令和回答按固定格式拼成一条序列，
           并在回答末尾放一个 <|endoftext|>(EOS)，教模型"答完就停"。
  【SFT-3】loss mask：整条序列都喂进去，但**只对"回答"那一段算 loss**，
           指令/模板那一段的标签设成 -100（被 cross_entropy 自动忽略）。
           ——这样模型学的是"给定指令该怎么答",而不是"怎么生成指令"。

训练循环本身几乎就是 04 的翻版：zero_grad → forward(算 loss) → backward → step。
唯一的差别就是上面那个 -100 掩码。

为了"看见"效果，脚本在训练**前**和训练**后**各采样一次同样的指令：
  - 训练前(base)：不认模板、跑题、**停不下来**（一直续写到 max_new_tokens）。
  - 训练后(SFT)：按模板给出"回答形状"，并在 EOS 处**主动停下**。

⚠️ 这是**玩具级**演示：124M 很小、只见过英文、训练数据只有十几条。
   目标是"看懂 SFT 机制 + 看到行为变化"，不是训出一个能真聊天的助手。
   想要能用的助手得换更大、且预训练含中文的开源底座——那是另一条线。

用法（本地 MPS / CPU 都能跑，几分钟）：
  python 06_sft.py --ckpt ../phase1-124m/ckpt10b/latest.pt --epochs 30
  python 06_sft.py --ckpt ../phase1-124m/ckpt300m/latest.pt --epochs 30 --lr 3e-5
"""
import os
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, default="../phase1-124m/ckpt10b/latest.pt",
                help="Phase 1 训出来的 base checkpoint")
ap.add_argument("--out_dir", type=str, default="ckpt_sft")
ap.add_argument("--epochs", type=int, default=30, help="把这十几条样本过多少遍")
ap.add_argument("--lr", type=float, default=2e-5, help="SFT 学习率：比预训练(6e-4)小 1~2 个量级")
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--max_new_tokens", type=int, default=64, help="采样时最多生成多少 token")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top_k", type=int, default=40)
ap.add_argument("--seed", type=int, default=1337)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# 模型定义：与 04 / 05 完全一致，这样 base 的 state_dict 键才对得上
# （训练版 forward：给 targets 就用 cross_entropy 算 loss，默认 ignore_index=-100）
# ---------------------------------------------------------------------------
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
        self.transformer.wte.weight = self.lm_head.weight   # 权重绑定（与 base 一致）
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
            # 默认 ignore_index=-100：标成 -100 的位置（=指令段）自动不算 loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ---------------------------------------------------------------------------
# 分词器 + 对话模板 + EOS
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
EOS = enc.eot_token   # <|endoftext|> = 50256，既是 GPT-2 的文末符，这里当 SFT 的"回答结束"标记

# 【SFT-2】对话模板：用固定文字标出"指令段"和"回答段"的边界。
# 标记用什么文字不重要，重要的是"全程一致"。真实 chat 模型常用 <|user|>/<|assistant|>，
# 这里用 Alpaca 风格的纯文字模板，好处是不必给词表新增特殊 token（vocab 保持 50304、权重绑定不变）。
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# ---------------------------------------------------------------------------
# 【SFT-1】数据：十几条「指令 → 回答」成对数据（玩具级，全英文，刻意短）
# ---------------------------------------------------------------------------
DATA = [
    ("Translate to French: I love cats.", "J'aime les chats."),
    ("Translate to French: Good morning.", "Bonjour."),
    ("Translate to French: Thank you very much.", "Merci beaucoup."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is 2 + 2?", "2 + 2 = 4."),
    ("What is 7 times 6?", "7 times 6 = 42."),
    ("Give me three primary colors.", "The three primary colors are red, blue, and yellow."),
    ("Name three fruits.", "Three fruits are apple, banana, and orange."),
    ("Write a short greeting.", "Hello! It is nice to meet you."),
    ("Define the word 'ocean'.", "An ocean is a very large body of salt water."),
    ("Rewrite politely: give me the report.", "Could you please send me the report?"),
    ("Is the sun a star? Answer yes or no.", "Yes, the sun is a star."),
    ("Complete: The opposite of hot is", "The opposite of hot is cold."),
    ("List two planets.", "Two planets are Earth and Mars."),
]

def build_example(instruction, response):
    """把一对 (指令, 回答) 拼成 (x, y),并按 SFT 规则做 loss mask。
       返回的 x、y 都是 python list,长度相同。"""
    prompt_ids = enc.encode_ordinary(PROMPT_TEMPLATE.format(instruction=instruction))
    resp_ids = enc.encode_ordinary(response) + [EOS]      # 回答末尾加 EOS,教它停
    full = prompt_ids + resp_ids
    # 【SFT-3】loss mask：指令段标 -100,回答段用真实 token id
    labels = [-100] * len(prompt_ids) + resp_ids
    # 标准 next-token 错位：用第 i 个 token 预测第 i+1 个
    x = full[:-1]
    y = labels[1:]
    return x, y

def make_batches(data, batch_size):
    """把样本切成若干 batch,每个 batch 内右侧补齐到等长
       (输入补 EOS、标签补 -100,使补位不产生梯度)。"""
    examples = [build_example(i, r) for i, r in data]
    batches = []
    for s in range(0, len(examples), batch_size):
        chunk = examples[s:s + batch_size]
        maxlen = max(len(x) for x, _ in chunk)
        xb, yb = [], []
        for x, y in chunk:
            pad = maxlen - len(x)
            xb.append(x + [EOS] * pad)        # 输入补位:补什么都行,反正它的 loss 被 mask
            yb.append(y + [-100] * pad)       # 标签补位:-100,补位不算 loss
        xb = torch.tensor(xb, dtype=torch.long, device=device)
        yb = torch.tensor(yb, dtype=torch.long, device=device)
        batches.append((xb, yb))
    return batches

# ---------------------------------------------------------------------------
# 采样:给一条指令,套上模板,生成回答(碰到 EOS 就停)
# ---------------------------------------------------------------------------
@torch.no_grad()
def chat(model, instruction, max_new_tokens, temperature, top_k):
    model.eval()
    ids = enc.encode_ordinary(PROMPT_TEMPLATE.format(instruction=instruction))
    x = torch.tensor(ids, dtype=torch.long, device=device)[None]
    out = []
    stopped = False
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -model.cfg.block_size:])
        logits = logits[:, -1, :]
        logits[:, 50257:] = float("-inf")      # 屏蔽词表里没训过的对齐空行(但保留 50256=EOS)
        logits = logits / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        if nxt.item() == EOS:                  # 模型主动说"答完了" → 停
            stopped = True
            break
        out.append(nxt.item())
        x = torch.cat([x, nxt], dim=1)
    text = enc.decode(out)
    tag = "⟨EOS,主动停⟩" if stopped else "⟨到长度上限被截断,没停下来⟩"
    return text, tag

EVAL_PROMPTS = [
    "What is the capital of France?",     # 训练里见过
    "Translate to French: I love cats.",  # 训练里见过
    "What is the capital of Germany?",    # 没见过,看它能否套用格式(玩具级,别强求答对)
]

def show_samples(model, title):
    print(f"\n========== {title} ==========")
    for p in EVAL_PROMPTS:
        torch.manual_seed(args.seed)
        text, tag = chat(model, p, args.max_new_tokens, args.temperature, args.top_k)
        print(f"\n[指令] {p}")
        print(f"[回答] {text.strip()}  {tag}")

# ---------------------------------------------------------------------------
# 主流程：加载 base → 采样(前) → SFT → 采样(后) → 存盘
# ---------------------------------------------------------------------------
def main():
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device} | 加载 base: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"base 来自 step {ckpt.get('step','?')}, val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # ---- 训练前:base 的样子(跑题、停不下来)----
    show_samples(model, "训练前 · base 模型(只会续写)")

    # ---- SFT 训练循环(就是 04 的循环 + loss mask)----
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    print(f"\n开始 SFT | 样本 {len(DATA)} 条 | epochs {args.epochs} | lr {args.lr} | batch {args.batch_size}")
    for ep in range(args.epochs):
        batches = make_batches(DATA, args.batch_size)   # 每个 epoch 重切(可加 shuffle,这里从简)
        ep_loss, nb = 0.0, 0
        for xb, yb in batches:
            opt.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)        # loss 只来自"回答段"(指令段是 -100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d} | 回答段 loss {ep_loss/nb:.4f}")

    # ---- 训练后:SFT 的样子(按格式答、EOS 处停)----
    show_samples(model, "训练后 · SFT 模型(会听指令、会停)")

    # ---- 存盘 ----
    out = os.path.join(args.out_dir, "sft.pt")
    torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                "base_ckpt": args.ckpt, "epochs": args.epochs}, out)
    print(f"\n✅ SFT 完成,已存到 {out}")
    print("对照上面【训练前 vs 训练后】:重点看回答是否套上了格式、是否在 ⟨EOS⟩ 处停下。")

if __name__ == "__main__":
    main()
