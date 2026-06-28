"""
里程碑 1+2：字符级 tokenizer + Bigram 基线模型
============================================

这是整条线的 "hello world"。目标不是好，是让你看清楚一个语言模型最朴素的样子：
  "给定当前这一个字符，预测下一个字符是什么。"

Bigram = 只看前一个 token。它没有注意力、没有上下文，蠢得很——
但它能跑、能训、能采样，并给我们一个 loss 基线（后面加注意力要打败它）。

makemore 的核心教训就藏在这里：语言模型 = 一个在词表上输出概率分布的分类器。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- 0. 设备 & 随机种子 ----------------------------------------------------
# 三平台通用：有 N 卡用 cuda（Windows/Linux），Mac 用 mps（Apple GPU），都没有就退回 cpu。
# 这个玩具脚本很小，CPU 也跑得动，只是慢一点——所以任何机器都能复现。
# 固定 seed 让每次结果可复现，方便我们讨论同一个数字。
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)
print(f"device = {device}")

# ---- 1. 读数据 & 构建字符级 tokenizer --------------------------------------
with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 词表 = 文本里出现过的所有不同字符，排序后固定下来。
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"vocab_size = {vocab_size}")           # 65：大小写字母 + 标点 + 换行等
print(f"chars = {''.join(chars)!r}")

# tokenizer 就是两张查找表：字符<->整数。
# 这就是 "tokenization" 最朴素的形态——一个字符一个 token，没有 BPE，没有子词。
stoi = {ch: i for i, ch in enumerate(chars)}   # string  -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int     -> string
encode = lambda s: [stoi[c] for c in s]        # "hi" -> [46, 47]
decode = lambda l: "".join(itos[i] for i in l) # [46, 47] -> "hi"

# 把整本书编码成一个长整数张量。
data = torch.tensor(encode(text), dtype=torch.long)

# 训练/验证集划分：前 90% 训练，后 10% 验证（用来发现过拟合）。
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---- 2. 取一个 batch -------------------------------------------------------
batch_size = 32        # 一次并行处理多少个序列
block_size = 8         # 上下文长度：bigram 其实只用最后 1 个，但我们按通用格式准备数据

# ---- 形状速查：代码里张量都按 (B, T, C) 三个大写字母标注 ----
#   B = Batch     一个 batch 有几条序列（并行）       = batch_size = 32
#   T = Time      一条序列有几个位置 / token          = block_size = 8
#   C = Channels  每个位置的向量有几个数（特征/通道）  本文件 = vocab_size = 65（logits 直接就是打分）

def get_batch(split):
    d = train_data if split == "train" else val_data
    # 随机选 batch_size 个起点
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])         # 输入：第 i..i+7 个字符
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix]) # 目标：每个位置的"下一个"字符
    return x.to(device), y.to(device)

# ---- 3. Bigram 模型 --------------------------------------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 一张 (vocab_size, vocab_size) 的表。
        # 第 i 行 = "看到 token i 后，下一个 token 的得分分布(logits)"。
        # 这就是整个模型——没有别的参数。
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) 一批 token 序列
        logits = self.token_embedding_table(idx)   # (B, T, vocab_size)：每个位置预测下一个

        if targets is None:
            return logits, None

        # 交叉熵要 (N, C) 的 logits 和 (N,) 的目标，所以把 B、T 摊平。
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx: (B, T) 已有的上下文；每次预测下一个字符，拼上去，循环。
        for _ in range(max_new_tokens):
            logits, _ = self(idx)            # (B, T, C)
            logits = logits[:, -1, :]        # 只要最后一个位置的预测 (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率采样，不取 argmax
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = BigramLanguageModel(vocab_size).to(device)
print(f"参数量 = {sum(p.numel() for p in model.parameters())}")   # 65*65 = 4225

# ---- 4. 训练前 loss 的"理论值" ---------------------------------------------
# 随机初始化时，模型对 65 个字符没有任何偏好 -> 均匀分布。
# 交叉熵 = -ln(1/65) = ln(65) ≈ 4.17。训练后应明显低于这个数。
print(f"理论初始 loss ≈ ln(vocab_size) = {torch.log(torch.tensor(vocab_size)).item():.4f}")

# ---- 5. 训练循环 -----------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(200)
        for k in range(200):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

max_iters = 3000
for it in range(max_iters):
    if it % 500 == 0 or it == max_iters - 1:
        l = estimate_loss()
        print(f"step {it:4d} | train loss {l['train']:.4f} | val loss {l['val']:.4f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---- 6. 采样：让训练后的模型"说话" -----------------------------------------
print("\n----- 采样结果（从换行符 0 开始，生成 500 字）-----")
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, max_new_tokens=500)[0].tolist()))
