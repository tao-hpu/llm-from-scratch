"""
里程碑 7：手搓 LoRA —— 冻结底座，只训一条"低秩旁路"，做同一件 SFT 的事
========================================================================

接着第 6 关。06 是**全量 SFT**：124M 个参数全部参与更新。
这一关换一种"手段"：**LoRA（低秩适配）**——任务还是同一个 SFT（让 base 会答题），
但**只训练大约 1% 的参数**，存盘的也只是一个几 MB 的小 adapter。

一句话抓住 LoRA：
    原来一层是   y = W · x                （W 是冻住、不动的底座权重）
    LoRA 改成    y = W · x + (B · A) · x   （只训练新加的 A、B 两个小矩阵）

为什么"低秩"省参数：设这一层 in=out=768。
  - 全量微调要动 W，规模 768 × 768 ≈ 59 万个数。
  - LoRA 只加 A(768×r) 和 B(r×768)。取 r=8，A+B 只有 768×8×2 ≈ 1.2 万个数，
    约为原来的 2%。把所有层加起来，可训练参数也就 ~1% 量级。

三处关键设计（代码里都标了【LoRA】）：
  【LoRA-1】冻结：加载完 base 后，把**所有**原参数 requires_grad=False。
           底座一个数都不变，只有新加的 A、B 会被优化器更新。
  【LoRA-2】旁路 + 初始化：每个目标 Linear 旁边并联 B·A。
           A 用小随机数、**B 置零** → 训练第 0 步旁路输出恒为 0，
           所以"装上 LoRA 的模型"一开始和 base 一模一样，是从 base 平滑出发的。
           缩放 scaling = alpha / r，用来调旁路的"音量"。
  【LoRA-3】只存 adapter：存盘只保存 A、B（几 MB），不保存底座。
           用的时候：加载原始 base + 叠上这个 adapter 即可，一份底座能挂多个 adapter。

除了"只训 A、B"，**训练循环、loss mask、对话模板、EOS 全部和 06 一字不差**——
LoRA 是"换了套要更新的参数"，不是"换了套学习目标"。

⚠️ 仍是**玩具级**演示（124M、纯英文、十几条数据）：目的是看懂 LoRA 机制 +
   看到"只动 1% 参数也能把 base 调成会答题"，不是训一个能真聊天的助手。

用法（本地 MPS / CPU 都能跑，几分钟）：
  python 07_lora.py --ckpt ../phase1-124m/ckpt10b/latest.pt --epochs 60
  python 07_lora.py --ckpt ../phase1-124m/ckpt10b/latest.pt --rank 4 --alpha 8
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
ap.add_argument("--out_dir", type=str, default="ckpt_lora")
ap.add_argument("--rank", type=int, default=8, help="LoRA 的秩 r：旁路的'宽度'，越小越省参数")
ap.add_argument("--alpha", type=float, default=16.0, help="LoRA 缩放，scaling = alpha / rank")
ap.add_argument("--epochs", type=int, default=60, help="LoRA 参数少，通常比全量 SFT 多过几遍")
ap.add_argument("--lr", type=float, default=1e-3, help="LoRA 学习率：通常比全量 SFT(2e-5) 大 1~2 个量级")
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--max_new_tokens", type=int, default=64)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top_k", type=int, default=40)
ap.add_argument("--seed", type=int, default=1337)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(args.seed)
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "10"))   # 每多少个 epoch 打印一次 loss

# ---------------------------------------------------------------------------
# 模型定义：与 04 / 05 / 06 完全一致，base 的 state_dict 键才对得上
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
        self.transformer.wte.weight = self.lm_head.weight
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

# ===========================================================================
# 【LoRA 核心】一个手搓的 LoRALinear：包住原 Linear，并联一条低秩旁路 B·A
# ===========================================================================
class LoRALinear(nn.Module):
    """y = base(x) + scaling * B(A(x))
       - base 是原来那层，整段冻结（不更新）
       - A: in_features → r   （小随机数初始化）
       - B: r → out_features  （置零初始化 → 初始旁路恒为 0）
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():       # 【LoRA-1】冻结原层
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.rank = rank
        self.scaling = alpha / rank             # 旁路"音量"
        # 【LoRA-2】旁路两块小矩阵（用 Linear 表示，无 bias）
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1.0 / rank)  # A：小随机
        nn.init.zeros_(self.lora_B.weight)                   # B：置零 → 初始 ΔW=0
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.drop(x)))


def inject_lora(model, rank, alpha):
    """把每个 Block 里的 4 个 Linear（注意力 c_attn/c_proj、MLP c_fc/c_proj）
       换成 LoRALinear。lm_head 与 wte 绑定、属底座，不加旁路。"""
    n = 0
    for blk in model.transformer.h:
        blk.attn.c_attn = LoRALinear(blk.attn.c_attn, rank, alpha)
        blk.attn.c_proj = LoRALinear(blk.attn.c_proj, rank, alpha)
        blk.mlp.c_fc    = LoRALinear(blk.mlp.c_fc,    rank, alpha)
        blk.mlp.c_proj  = LoRALinear(blk.mlp.c_proj,  rank, alpha)
        n += 4
    return n


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

# ---------------------------------------------------------------------------
# 分词器 + 对话模板 + EOS（与 06 完全一致）
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
EOS = enc.eot_token
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# ---------------------------------------------------------------------------
# 数据 / 拼样本 / 批次：与 06 完全相同（同一个 SFT 任务，方便横向对比）
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
    prompt_ids = enc.encode_ordinary(PROMPT_TEMPLATE.format(instruction=instruction))
    resp_ids = enc.encode_ordinary(response) + [EOS]
    full = prompt_ids + resp_ids
    labels = [-100] * len(prompt_ids) + resp_ids   # loss mask：指令段不算 loss
    return full[:-1], labels[1:]

def make_batches(data, batch_size):
    examples = [build_example(i, r) for i, r in data]
    batches = []
    for s in range(0, len(examples), batch_size):
        chunk = examples[s:s + batch_size]
        maxlen = max(len(x) for x, _ in chunk)
        xb, yb = [], []
        for x, y in chunk:
            pad = maxlen - len(x)
            xb.append(x + [EOS] * pad)
            yb.append(y + [-100] * pad)
        xb = torch.tensor(xb, dtype=torch.long, device=device)
        yb = torch.tensor(yb, dtype=torch.long, device=device)
        batches.append((xb, yb))
    return batches

# ---------------------------------------------------------------------------
# 采样（与 06 一致）
# ---------------------------------------------------------------------------
@torch.no_grad()
def chat(model, instruction, max_new_tokens, temperature, top_k):
    model.eval()
    ids = enc.encode_ordinary(PROMPT_TEMPLATE.format(instruction=instruction))
    x = torch.tensor(ids, dtype=torch.long, device=device)[None]
    out, stopped = [], False
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -model.cfg.block_size:])
        logits = logits[:, -1, :]
        logits[:, 50257:] = float("-inf")
        logits = logits / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        if nxt.item() == EOS:
            stopped = True
            break
        out.append(nxt.item())
        x = torch.cat([x, nxt], dim=1)
    tag = "⟨EOS,主动停⟩" if stopped else "⟨到长度上限被截断,没停下来⟩"
    return enc.decode(out), tag

EVAL_PROMPTS = [
    "What is the capital of France?",
    "Translate to French: I love cats.",
    "What is the capital of Germany?",
]

def show_samples(model, title):
    print(f"\n========== {title} ==========")
    for p in EVAL_PROMPTS:
        torch.manual_seed(args.seed)
        text, tag = chat(model, p, args.max_new_tokens, args.temperature, args.top_k)
        print(f"\n[指令] {p}")
        print(f"[回答] {text.strip()}  {tag}")

# ---------------------------------------------------------------------------
# 主流程：加载 base → 冻结 → 注入 LoRA → 采样(前) → 只训 A/B → 采样(后) → 只存 adapter
# ---------------------------------------------------------------------------
def main():
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device} | 加载 base: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])     # 先按原结构加载,key 才对得上
    print(f"base 来自 step {ckpt.get('step','?')}, val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # 【LoRA-1】把底座全部冻结，再注入旁路（旁路参数默认 requires_grad=True）
    for p in model.parameters():
        p.requires_grad = False
    n_lora = inject_lora(model, args.rank, args.alpha)
    model.to(device)
    trainable, total = count_params(model)
    print(f"注入 LoRA：{n_lora} 个 Linear 加了旁路 | r={args.rank} alpha={args.alpha} scaling={args.alpha/args.rank:g}")
    print(f"可训练参数 {trainable:,} / 总参数 {total:,}  =  {100*trainable/total:.2f}%")

    # ---- 训练前：装了 LoRA 但 B=0，输出应与 base 一致（跑题、停不下来）----
    show_samples(model, "训练前 · base + 空 LoRA(B=0，等价 base)")

    # ---- 只把"需要梯度"的参数(=A、B)交给优化器 ----
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    print(f"\n开始 LoRA 微调 | 样本 {len(DATA)} 条 | epochs {args.epochs} | lr {args.lr} | batch {args.batch_size}")
    model.train()
    for ep in range(args.epochs):
        batches = make_batches(DATA, args.batch_size)
        ep_loss, nb = 0.0, 0
        for xb, yb in batches:
            opt.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)                 # loss 只来自回答段(指令段 -100)
            loss.backward()                          # 梯度只流向 A、B，底座不动
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        if ep % PRINT_EVERY == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d} | 回答段 loss {ep_loss/nb:.4f}")

    # ---- 训练后：只动了 1% 参数，应该也学会了按格式答、EOS 处停 ----
    show_samples(model, "训练后 · LoRA(只训了 A、B)")

    # ---- 【LoRA-3】只存 adapter（含 A、B 的权重），不存底座 ----
    lora_sd = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    out = os.path.join(args.out_dir, "lora.pt")
    torch.save({"lora": lora_sd, "rank": args.rank, "alpha": args.alpha,
                "base_ckpt": args.ckpt, "epochs": args.epochs}, out)
    adapter_mb = os.path.getsize(out) / 1e6
    full_mb = os.path.getsize(args.ckpt) / 1e6
    print(f"\n✅ LoRA 完成，只存 adapter 到 {out}")
    print(f"   adapter 大小 ≈ {adapter_mb:.1f} MB（对比完整 base ≈ {full_mb:.0f} MB）")
    print("用法：加载原始 base + 叠上这个 adapter 即可。一份底座可挂多个不同 adapter。")
    print("对照【训练前 vs 训练后】：只动 1% 参数，base 同样被调成了会答题、会停。")

if __name__ == "__main__":
    main()
