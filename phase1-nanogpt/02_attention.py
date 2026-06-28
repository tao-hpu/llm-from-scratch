"""
里程碑 3：单头自注意力（self-attention）
========================================

bigram 的墙是 2.47，因为它只能看前 1 个字符。
这一步给模型装上"往回看"的能力 —— 自注意力。

文件分两段：
  A 段：用 3x3 小例子，让你"看穿"注意力的数学本质（不训练，纯演示）
  B 段：真正的单头自注意力语言模型，训练并采样，打败 2.47

注意：和上一份对比，数据管线 get_batch / 训练循环 / generate 几乎没动，
      变的只有中间那个 class。这就是我上一步埋伏笔的兑现。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# 三平台通用：N 卡 cuda → Apple mps → cpu，都没有也能跑（小模型，CPU 慢点而已）。
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)
print(f"device = {device}")

# ============================================================================
# A 段：注意力的数学本质 —— "对过去做加权平均"
# ============================================================================
# 核心直觉：第 t 个位置想"看过去"，最朴素的做法 = 把第 0..t 个位置的向量平均一下。
# "只看过去、不看未来"，靠一个下三角矩阵实现。

print("\n===== A 段：用矩阵乘法实现'只对过去做平均' =====")
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))   # 下三角全 1
a = a / a.sum(dim=1, keepdim=True) # 每行归一化 -> 每行是一组"权重"，和为 1
b = torch.arange(6, dtype=torch.float).view(3, 2)  # 3 个位置，每个是 2 维向量
c = a @ b                          # 矩阵乘法 = 加权求和

print("权重矩阵 a（下三角归一化）=\n", a)
print("\n输入 b（3 个位置的向量）=\n", b)
print("\n输出 c = a @ b =\n", c)
print("""
解读：
  c 第 0 行 = b 第 0 行              （位置 0 只能看自己）
  c 第 1 行 = (b0 + b1) / 2          （位置 1 看 0,1 的平均）
  c 第 2 行 = (b0 + b1 + b2) / 3     （位置 2 看 0,1,2 的平均）
下三角 = 因果掩码(causal mask)：未来的位置权重是 0，谁也偷看不到答案。

自注意力做的事一模一样，只是把"平均"(固定权重)换成
"由内容算出来的权重" —— 谁和我相关，我就多看谁。下面 B 段就是这个升级。
""")

# ============================================================================
# B 段：真正的单头自注意力语言模型
# ============================================================================

# ---- 数据（和 01_bigram.py 完全一样，原样搬过来）----
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

# ---- 超参数 ----
batch_size = 32
block_size = 8       # 上下文长度：现在真的用上了，模型能看最多前 8 个字符
n_embd = 32          # 每个 token 嵌入成 32 维向量（bigram 时是 65 维 logits，没有"语义空间"）
head_size = 32       # 注意力头的维度（单头，设成和 n_embd 一样）
lr = 1e-3
max_iters = 5000

# ---- 形状速查：代码里张量都按 (B, T, C) 三个大写字母标注 ----
#   B = Batch     一个 batch 有几条序列（并行）       = batch_size = 32
#   T = Time      一条序列有几个位置 / token          = block_size = 8
#   C = Channels  每个位置的向量有几个数（特征/通道），随“是哪一步的张量”而变：
#                 嵌入后 x、q/k/v、out 的 C = n_embd / head_size = 32；最后 logits 的 C = vocab_size = 65

def get_batch(split):  # 和上一份一字不差
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---- 单个注意力头 ----
class Head(nn.Module):
    """一个自注意力头：每个位置发出 query，向所有过去位置的 key 打分，再按分加权聚合 value。"""
    def __init__(self, head_size):
        super().__init__()
        # 三个线性投影，没有 bias。它们把同一个输入向量投影成三种角色：
        self.key = nn.Linear(n_embd, head_size, bias=False)    # "我有什么内容"（被查询）
        self.query = nn.Linear(n_embd, head_size, bias=False)  # "我在找什么"（发起查询）
        self.value = nn.Linear(n_embd, head_size, bias=False)  # "如果你看我，我给你什么信息"
        # tril 不是参数（不训练），用 register_buffer 挂在模块上，随 .to(device) 一起搬。
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)

        # 1) 算注意力分数：每个 query 和每个 key 做点积。点积大 = 相关性高。
        wei = q @ k.transpose(-2, -1)              # (B, T, T)：wei[b,i,j] = 位置 i 对位置 j 的关注分
        # 2) 缩放：除以 sqrt(head_size)，防止点积过大把 softmax 推到极端（梯度消失）。
        wei = wei * (head_size ** -0.5)
        # 3) 因果掩码：未来位置设成 -inf，softmax 后权重变 0 —— 这就是 A 段那个下三角。
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # 4) softmax：把每一行分数变成和为 1 的权重分布。
        wei = F.softmax(wei, dim=-1)               # (B, T, T)

        # 5) 用权重去加权聚合 value（这一步对应 A 段的 a @ b）。
        v = self.value(x)    # (B, T, head_size)
        out = wei @ v        # (B, T, head_size)
        return out

# ---- 模型 ----
class AttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 现在 token 嵌入到 n_embd 维"语义空间"，不再直接是 logits。
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 位置嵌入：注意力本身不区分顺序，必须额外告诉模型"这是第几个位置"。
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(head_size)                    # 单个自注意力头
        self.lm_head = nn.Linear(head_size, vocab_size)   # 把聚合后的向量映射回词表得分

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                  # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T,n_embd)
        x = tok_emb + pos_emb       # 内容信息 + 位置信息，相加
        x = self.sa_head(x)         # (B,T,head_size)：每个位置已经"看过"它的过去
        logits = self.lm_head(x)    # (B,T,vocab_size)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 关键区别：注意力只能处理 <= block_size 的上下文，必须裁掉过早的部分。
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = AttentionLanguageModel().to(device)
print(f"\n参数量 = {sum(p.numel() for p in model.parameters())}")

# ---- 训练（循环结构和上一份完全一样）----
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

print("\n===== B 段：训练单头自注意力 =====")
for it in range(max_iters):
    if it % 1000 == 0 or it == max_iters - 1:
        l = estimate_loss()
        print(f"step {it:4d} | train loss {l['train']:.4f} | val loss {l['val']:.4f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n----- 采样结果（生成 500 字）-----")
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, max_new_tokens=500)[0].tolist()))
