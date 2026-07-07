"""
里程碑 5：手搓 MoE —— 把 FFN 换成"多个专家 + 路由器"
====================================================

03 的 GPT 里,FFN 是"每个 token 都要过的同一个小 MLP":参数全员上岗,一个都不能少。
MoE(Mixture of Experts,专家混合)把这一块拆开:

  1. Expert   把一个大 FFN 换成 n_expert 个小 FFN("专家"),各自学各自擅长的加工方式
  2. Router   一个线性层给每个 token 打分:这个 token 该找哪几个专家
  3. Top-k    每个 token 只走分数最高的 top_k 个专家 —— 参数变多了,每个 token 的计算量却没变
  4. 负载均衡  路由器天生爱偏心(全塞给一两个专家 = "专家塌缩"),加一项 aux loss 逼它雨露均沾

这正是 Mixtral / DeepSeek / Qwen-MoE 这代模型"参数千亿、激活百亿"的核心机关。
注意力/残差/LayerNorm/训练循环全部原封不动 —— 唯一变化:Block 里的 FeedForward 换成 MoELayer。

本份的"对照组"就是 03_transformer.py(dense):
  - dense  FFN 隐层 4*n_embd,          每 token 激活 = 全部 FFN 参数
  - MoE    4 个专家、每个隐层 2*n_embd、top_k=2 → 总 FFN 参数是 dense 的 2 倍,
           但每个 token 只走 2 个专家,激活的 FFN 计算量和 dense 完全一样(2 × 2*n_embd = 4*n_embd)
  → 同样的"每 token 算力"买到了双倍的"容量",这就是 MoE 的交易。

跑法:python3 09_moe.py            (训 MoE)
     python3 09_moe.py --aux 0    (关掉负载均衡,亲眼看专家塌缩)
预期:val loss ≈ 1.55(略优于 dense 的 ~1.57);训练结尾打印每个专家的
     负载占比 + 最常接手的字符,能看到专家真的各有分工。
"""

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- 超参数(在 03 的 Mac 友好版之上,只加 MoE 三个)----
batch_size = 64
block_size = 64       # 上下文长度
n_embd = 128          # 嵌入维度
n_head = 4            # 注意力头数;每个头维度 = n_embd / n_head = 32
n_layer = 4           # 堆几层 Block
dropout = 0.1

n_expert = 4          # 专家个数(每层 4 个小 FFN)
top_k = 2             # 每个 token 走几个专家(Mixtral 同款 top-2)
expert_hidden = 2 * n_embd  # 每个专家的隐层宽度。top_k=2 × 2*n_embd = 4*n_embd,
                            # 每 token 激活的 FFN 算力 = dense 版,公平对照

lr = 1e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200

parser = argparse.ArgumentParser()
parser.add_argument("--aux", type=float, default=0.01,
                    help="负载均衡 loss 系数(默认 0.01;设 0 可观察专家塌缩)")
parser.add_argument("--max-iters", type=int, default=max_iters)
args = parser.parse_args()
aux_coef = args.aux
max_iters = args.max_iters

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)
print(f"device = {device} | n_expert = {n_expert} | top_k = {top_k} | aux_coef = {aux_coef}")

# ---- 数据(还是 tiny shakespeare,复用 phase1 的文件)----
with open("../phase1-nanogpt/data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
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

# ---- 注意力(和 03 完全一样,MoE 不碰这里)----
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
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

# ---- 1. Expert:一个"变窄了"的 FFN ----
class Expert(nn.Module):
    """结构和 03 的 FeedForward 一模一样,只是隐层从 4*n_embd 缩到 expert_hidden。
    一个专家单干不如 dense FFN;MoE 靠"派对专家"把它们拼回来。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expert_hidden),
            nn.ReLU(),
            nn.Linear(expert_hidden, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ---- 2+3+4. MoELayer:router 打分 → top-k 派单 → 加权汇总 + 负载均衡 ----
class MoELayer(nn.Module):
    """替换 Block 里的 FeedForward。逐 token 决策:每个 token 独立选自己的专家
    (同一句话里相邻两个字可以走完全不同的专家 —— 路由的单位是 token,不是句子)。"""
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(n_expert)])
        self.router = nn.Linear(n_embd, n_expert, bias=False)  # 路由器就一个线性层
        # 统计用:训练外也能看每个专家接了多少 token
        self.register_buffer("usage", torch.zeros(n_expert), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        flat = x.view(B * T, C)                       # 路由按 token 做,先摊平

        logits = self.router(flat)                    # (B*T, n_expert) 每个 token 给每个专家打分
        probs = F.softmax(logits, dim=-1)             # 变成"该派给谁"的概率
        topv, topi = probs.topk(top_k, dim=-1)        # 只留分数最高的 top_k 个专家
        topv = topv / topv.sum(dim=-1, keepdim=True)  # 在选中的 k 个里重新归一化,权重和=1

        out = torch.zeros_like(flat)
        for e in range(n_expert):                     # 教学版:按专家循环,清晰优先
            mask = (topi == e)                        # 哪些 (token, 槽位) 派给了专家 e
            if not mask.any():
                continue
            tok_idx = mask.any(dim=-1).nonzero(as_tuple=True)[0]   # 被派到 e 的 token 行号
            w = (topv * mask).sum(dim=-1)[tok_idx].unsqueeze(-1)   # 对应权重
            out[tok_idx] += w * self.experts[e](flat[tok_idx])     # 专家只算自己名下的 token

        # ---- 负载均衡 aux loss(Switch Transformer 式)----
        # f_e = 实际派给专家 e 的 token 占比;P_e = 路由器给 e 的平均概率。
        # loss = n_expert * Σ f_e·P_e:均匀时 = 1(最小),越偏心越大。
        # 只加它才能防"强者恒强"的正反馈 —— 被冷落的专家学不到东西,分数更低,更被冷落。
        f = mask_load = topi.view(-1).bincount(minlength=n_expert).float() / topi.numel()
        P = probs.mean(dim=0)
        self.aux_loss = n_expert * (f * P).sum()

        self.usage = mask_load.detach()               # 留给训练结尾打印
        return out.view(B, T, C)

class Block(nn.Module):
    """和 03 唯一的区别:self.ffwd 从 FeedForward 换成 MoELayer。残差、pre-norm 原样。"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = MoELayer()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # 沟通:token 之间交换信息(不变)
        x = x + self.ffwd(self.ln2(x))    # 思考:现在是"各找各的专家"思考
        return x

# ---- 完整 GPT(和 03 相同,多收集一项 aux loss)----
class MoEGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None, None
        B, T, C = logits.shape
        ce = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        aux = sum(b.ffwd.aux_loss for b in self.blocks) / n_layer   # 各层负载均衡取平均
        return logits, ce, aux

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = MoEGPT().to(device)
n_param = sum(p.numel() for p in model.parameters())
# 激活参数:非 FFN 部分全算 + 每层只走 top_k 个专家
n_ffn_total = sum(p.numel() for b in model.blocks for p in b.ffwd.experts.parameters())
n_active = n_param - n_ffn_total + n_ffn_total // n_expert * top_k
print(f"总参数 = {n_param/1e6:.2f} M | 每 token 激活 ≈ {n_active/1e6:.2f} M"
      f"(dense 对照 03 约 0.81 M 全激活)")

# ---- 训练(和 03 同一个循环,loss 多加一项 aux)----
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, ce, _ = model(*get_batch(split))
            losses[k] = ce.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def usage_report():
    """打印每层每个专家实际接单占比(完美均衡 = 每人 1/n_expert = 25%)。"""
    for li, b in enumerate(model.blocks):
        u = b.ffwd.usage
        bars = " | ".join(f"E{e}:{u[e]*100:4.1f}%" for e in range(n_expert))
        print(f"  layer {li}: {bars}")

print("===== 训练 MoE Transformer =====")
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        l = estimate_loss()
        u = model.blocks[0].ffwd.usage
        ub = "/".join(f"{v*100:.0f}" for v in u)
        print(f"step {it:4d} | train loss {l['train']:.4f} | val loss {l['val']:.4f}"
              f" | layer0 负载 {ub}%")
    xb, yb = get_batch("train")
    _, ce, aux = model(xb, yb)
    loss = ce + aux_coef * aux            # aux_coef=0 时路由器放飞自我 → 专家塌缩
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n----- 各层专家负载(最后一个 batch)-----")
usage_report()

# ---- 专家分工一瞥:每个专家最常接手哪些字符(看第 0 层)----
@torch.no_grad()
def specialization_report():
    model.eval()
    counts = torch.zeros(n_expert, vocab_size)
    for _ in range(20):
        xb, _ = get_batch("val")
        moe = model.blocks[0].ffwd
        x = model.token_embedding_table(xb) + model.position_embedding_table(
            torch.arange(xb.shape[1], device=device))
        x = model.blocks[0].ln2(x + model.blocks[0].sa(model.blocks[0].ln1(x)))
        top1 = moe.router(x.view(-1, n_embd)).argmax(dim=-1)      # 每个 token 的首选专家
        for e in range(n_expert):
            ids = xb.view(-1)[top1 == e]
            counts[e] += ids.cpu().bincount(minlength=vocab_size).float()
    print("\n----- 专家分工(layer 0,val 集,每专家最常接的字符)-----")
    for e in range(n_expert):
        top = counts[e].topk(8).indices.tolist()
        share = counts[e].sum() / counts.sum() * 100
        print(f"  Expert {e}({share:4.1f}% 单量): {[repr(itos[i]) for i in top]}")

    # 字符类别 × 专家:每类字符被派往各专家的占比(一行加起来 = 100%)
    cats = {
        "空白": set(" \n"),
        "元音": set("aeiou"),
        "辅音": set("bcdfghjklmnpqrstvwxyz"),
        "大写": set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "标点": set(".,:;!?'-"),
    }
    print("\n----- 字符类别 → 专家 路由占比(layer 0)-----")
    print("        " + "".join(f"E{e:<7d}" for e in range(n_expert)))
    for name, cs in cats.items():
        ids = [stoi[c] for c in cs if c in stoi]
        row = counts[:, ids].sum(dim=1)
        row = row / row.sum() * 100
        print(f"  {name}  " + "".join(f"{v:5.1f}%  " for v in row))
    model.train()

specialization_report()

print("\n----- 采样结果(生成 500 字)-----")
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, max_new_tokens=500)[0].tolist()))
