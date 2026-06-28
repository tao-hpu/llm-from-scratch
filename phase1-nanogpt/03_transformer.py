"""
里程碑 4：完整的 Transformer —— 真正的 GPT
==========================================

02 的单头注意力把 loss 从 2.49 压到 2.41，但那还是个"玩具"：1 个头、1 层、32 维。
这一步加四样东西，把它变成 Karpathy "Let's build GPT" 的终点，也就是 nanoGPT 的内核：

  1. Multi-Head    多个注意力头并行 —— 每个头关注不同模式（语法/主谓/标点…）
  2. FeedForward   逐位置的小 MLP —— 注意力负责"沟通"，FFN 负责"思考"
  3. 残差连接       让梯度能穿过深层 —— 没有它，堆深了根本训不动（你以后研究优化器会反复碰到）
  4. LayerNorm     稳定每层的激活分布 —— 让深网络的训练不发散

数据管线 / 训练循环 / generate 还是没怎么变。变的是模型从"一个 class"长成了"堆叠的 Block"。

跑法：python3 03_transformer.py
预期：val loss 从 2.41 掉到 ~1.9，采样开始有"词"和对话结构（不再是纯字母汤）。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- 超参数（Mac 友好版：几分钟跑完）----
# 想要"真·像样的莎士比亚"，把下面注释里的"4090 版"打开，丢到 4090 上跑，loss 能到 ~1.5。
batch_size = 64
block_size = 64       # 上下文长度（4090 版：256）
n_embd = 128          # 嵌入维度（4090 版：384）
n_head = 4            # 注意力头数（4090 版：6）；每个头维度 = n_embd / n_head = 32
n_layer = 4           # 堆几层 Block（4090 版：6）
dropout = 0.1         # 随机丢弃，防过拟合（4090 版：0.2）

# ---- 形状速查：代码里张量都按 (B, T, C) 三个大写字母标注 ----
#   B = Batch     一个 batch 有几条序列（并行）       = batch_size = 64
#   T = Time      一条序列有几个位置 / token          = block_size = 64
#   C = Channels  每个位置的向量有几个数（特征/通道），随“是哪一步的张量”而变：
#                 主干 x = n_embd = 128；单个头内部 q/k/v/out = head_size = n_embd//n_head = 32；最后 logits = vocab_size
lr = 1e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200

# 三平台通用：N 卡 cuda → Apple mps → cpu。这份稍大，有 GPU 会明显快很多。
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)
print(f"device = {device}")

# ---- 数据（和前两份一样）----
with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---- 单个注意力头（和 02 几乎一样，多了一个 dropout）----
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)      # (B,T,T) 打分+缩放
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # 因果掩码
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)                                    # 随机丢一些注意力连接
        return wei @ self.value(x)                                # 加权聚合 value

# ---- 1. Multi-Head：多个头并行，再拼起来 ----
class MultiHeadAttention(nn.Module):
    """把注意力拆成 num_heads 个小头，各看各的，最后拼接 + 投影回 n_embd。"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)   # 拼接后投影回主维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)    # 沿特征维拼接
        return self.dropout(self.proj(out))

# ---- 2. FeedForward：逐位置的小 MLP ----
class FeedForward(nn.Module):
    """注意力让 token 之间'交换信息'，FFN 让每个 token 单独'加工信息'。中间放大 4 倍是惯例。"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # 投影回主维度，方便残差相加
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ---- 3+4. Block：把注意力和 FFN 用"残差 + 预归一化"包起来 ----
class Block(nn.Module):
    """一层 Transformer = 注意力子层 + FFN 子层，各自带残差和 LayerNorm。"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)   # 注意力前归一化
        self.ln2 = nn.LayerNorm(n_embd)   # FFN 前归一化

    def forward(self, x):
        # 注意这里的 x = x + ...：这就是【残差连接】。
        # 子层只学"在原信号上加什么修正"，梯度可以沿着这条 + 的"高速公路"直通底层，
        # 所以才堆得动深网络。这是 2015 年 ResNet 的核心思想，Transformer 全靠它。
        # 先 LayerNorm 再进子层 = "pre-norm"，比原始 Transformer 的 post-norm 更好训。
        x = x + self.sa(self.ln1(x))      # 沟通：每个 token 看它的过去
        x = x + self.ffwd(self.ln2(x))    # 思考：每个 token 单独加工
        return x

# ---- 完整 GPT ----
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)                  # 最后一层归一化
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                               # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)        # 过 n_layer 层 Block
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]   # 注意力只能看 block_size，裁剪
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = GPT().to(device)
print(f"参数量 = {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# ---- 训练 ----
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

print("===== 训练完整 Transformer =====")
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        l = estimate_loss()
        print(f"step {it:4d} | train loss {l['train']:.4f} | val loss {l['val']:.4f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n----- 采样结果（生成 800 字）-----")
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, max_new_tokens=800)[0].tolist()))
