"""
里程碑 8：手搓 DPO —— 从"会答题"到"答得合人意"(偏好对齐,不走 RL)
========================================================================

这是 Phase 2 手搓三关的收尾。06 全量 SFT、07 LoRA 解决的都是同一件事:
让 base **会听指令答题**。但"会答"不等于"答得好"——同一个问题模型能给出好几种
都说得通的回答,我们想让它**更偏向人类更喜欢的那一种**(更礼貌 / 更简洁 / 更得体)。
这一步叫**偏好对齐**。

对齐有两条路(白皮书 step 5 讲过):
  - RLHF / PPO：先训一个奖励模型打分,再用强化学习去最大化分数。链路长、要采样。
  - DPO（本关）：**不训奖励模型、不走 RL**。直接拿成对的偏好数据
    (prompt, chosen=更好的回答, rejected=更差的回答),用一个简单的分类式 loss,
    把"模型对 chosen 的偏好"相对"对 rejected 的偏好"直接拉大。

DPO 的三个关键(代码里都标了【DPO】):
  【DPO-1】两个模型：
       - policy（要训的,从 SFT 模型出发）
       - ref（参考模型,SFT 模型的**冻结副本**,永不更新)
     "对齐"被定义成"相对参考模型的偏移",ref 当锚点,防止 policy 跑太偏。
  【DPO-2】隐式奖励 = β·(logπ_policy(y|x) − logπ_ref(y|x))。
     某条回答的"分数"不靠额外的奖励模型,而是"policy 比 ref 多给了它多少对数概率"。
     一条回答的 logπ 就是它每个 token 的 log-softmax 之和(只算 response 段,
     和 SFT 的 loss mask 同一个套路)。
  【DPO-3】loss = −log σ( β·[ (Δ_chosen) − (Δ_rejected) ] ),
     其中 Δ = logπ_policy − logπ_ref。
     直觉:让 chosen 的隐式奖励**高于** rejected,差距越大 loss 越小。
     这就是个"二选一,选对的"分类问题,所以不需要 RL。

横向看 Phase 2:**任务 × 手段是两个正交的轴**。
  - 任务:SFT(会答) / 偏好对齐(答得好)
  - 手段:全量微调 / LoRA
本关默认全量 DPO;加 --lora 就是"LoRA 手段 + DPO 任务",证明两轴可自由组合。

⚠️ 仍是**玩具级**(124M、十几条偏好对、纯英文)。在这么小的规模上,采样文字的变化
   往往很微妙;真正能"看见 DPO 起作用"的硬指标是:**偏好准确率↑、奖励 margin↑**
   (policy 越来越把 chosen 排在 rejected 前面)。脚本会把这些数都打出来。

用法(需要先有 06 的 SFT 产物 ckpt_sft/sft.pt):
  python 08_dpo.py --ckpt ckpt_sft/sft.pt --epochs 40
  python 08_dpo.py --ckpt ckpt_sft/sft.pt --epochs 60 --lora   # 改用 LoRA 手段做 DPO
"""
import os
import copy
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
ap.add_argument("--ckpt", type=str, default="ckpt_sft/sft.pt",
                help="DPO 接在 SFT 之后:这里给 06 训出来的 SFT checkpoint")
ap.add_argument("--out_dir", type=str, default="ckpt_dpo")
ap.add_argument("--beta", type=float, default=0.1, help="DPO 温度 β:控制相对参考模型能偏多远")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=5e-7, help="全量 DPO 用极小的 lr,防止 policy 偏离 ref 太远;--lora 时会自动放大")
ap.add_argument("--lora", action="store_true", help="改用 LoRA 手段做 DPO(冻底座只训旁路)")
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=16.0)
ap.add_argument("--max_new_tokens", type=int, default=64)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top_k", type=int, default=40)
ap.add_argument("--seed", type=int, default=1337)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(args.seed)
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "5"))

# ---------------------------------------------------------------------------
# 模型定义：与 04 / 05 / 06 / 07 完全一致
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

# ---------------------------------------------------------------------------
# LoRA(与 07 相同;仅当 --lora 时用到)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))

def inject_lora(model, rank, alpha):
    n = 0
    for blk in model.transformer.h:
        blk.attn.c_attn = LoRALinear(blk.attn.c_attn, rank, alpha)
        blk.attn.c_proj = LoRALinear(blk.attn.c_proj, rank, alpha)
        blk.mlp.c_fc    = LoRALinear(blk.mlp.c_fc,    rank, alpha)
        blk.mlp.c_proj  = LoRALinear(blk.mlp.c_proj,  rank, alpha)
        n += 4
    return n

# ---------------------------------------------------------------------------
# 分词器 + 模板 + EOS（与 06/07 一致）
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
EOS = enc.eot_token
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# ---------------------------------------------------------------------------
# 【DPO 数据】偏好三元组 (指令, chosen=更合人意, rejected=更差)
# 刻意做成"两个都答到了点上,但一个礼貌/简洁/得体,一个生硬/啰嗦/冒犯"
# ---------------------------------------------------------------------------
PREF = [
    ("What is the capital of France?",
     "The capital of France is Paris.",
     "Why are you asking me that? Look it up yourself."),
    ("Rewrite politely: give me the report.",
     "Could you please send me the report?",
     "Give me the report right now."),
    ("Write a short greeting.",
     "Hello! It is nice to meet you.",
     "What do you want?"),
    ("What is 2 + 2?",
     "2 + 2 = 4.",
     "That is a really stupid question, but fine, it is 4."),
    ("Name three fruits.",
     "Three fruits are apple, banana, and orange.",
     "Fruits fruits fruits, there are so many, like apple and banana and orange and grape and melon and..."),
    ("Define the word 'ocean'.",
     "An ocean is a very large body of salt water.",
     "An ocean is water. Just water. A lot of it. Water everywhere. Water water water."),
    ("Is the sun a star? Answer yes or no.",
     "Yes, the sun is a star.",
     "Well, it depends on how you define things, and honestly I am not sure I want to answer."),
    ("Give me three primary colors.",
     "The three primary colors are red, blue, and yellow.",
     "Colors? Ugh. Red and blue and yellow and also maybe others I cannot be bothered to list."),
    ("Translate to French: Thank you very much.",
     "Merci beaucoup.",
     "I don't really feel like translating that for you."),
    ("Complete: The opposite of hot is",
     "The opposite of hot is cold.",
     "The opposite of hot is, um, not hot, you figure it out."),
]

# ---------------------------------------------------------------------------
# 一条回答的总 log 概率(只算 response 段,和 SFT loss mask 同一道理)
# 返回标量 tensor;policy 需要梯度,ref 用 no_grad 包在外面。
# ---------------------------------------------------------------------------
def seq_logprob(model, instruction, response):
    prompt_ids = enc.encode_ordinary(PROMPT_TEMPLATE.format(instruction=instruction))
    resp_ids = enc.encode_ordinary(response) + [EOS]
    full = prompt_ids + resp_ids
    P = len(prompt_ids)
    x = torch.tensor(full[:-1], dtype=torch.long, device=device)[None]
    tgt = torch.tensor(full[1:], dtype=torch.long, device=device)        # (L-1,) 每个位置的真实下一 token
    logits, _ = model(x)
    logp = F.log_softmax(logits[0].float(), dim=-1)                      # (L-1, V)
    sel = logp.gather(1, tgt[:, None]).squeeze(1)                        # (L-1,) 真实 token 的 log 概率
    return sel[P - 1:].sum()                                             # 只累加"预测 response 段"那些位置

# ---------------------------------------------------------------------------
# 采样(与 06/07 一致)——DPO 前后各看一眼回答风格
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
    tag = "⟨EOS,主动停⟩" if stopped else "⟨没停,被截断⟩"
    return enc.decode(out), tag

EVAL_PROMPTS = [
    "Rewrite politely: give me the report.",
    "What is 2 + 2?",
    "Write a short greeting.",
]
def show_samples(model, title):
    print(f"\n========== {title} ==========")
    for p in EVAL_PROMPTS:
        torch.manual_seed(args.seed)
        text, tag = chat(model, p, args.max_new_tokens, args.temperature, args.top_k)
        print(f"\n[指令] {p}\n[回答] {text.strip()}  {tag}")

# ---------------------------------------------------------------------------
# 一遍数据上的 DPO loss + 诊断指标
# ---------------------------------------------------------------------------
def dpo_pass(policy, ref, train=True):
    total_loss, margin_sum, correct = 0.0, 0.0, 0
    rew_c_sum, rew_r_sum = 0.0, 0.0
    for instr, chosen, rejected in PREF:
        # 【DPO-2】policy / ref 对 chosen、rejected 的 log 概率
        pol_c = seq_logprob(policy, instr, chosen)
        pol_r = seq_logprob(policy, instr, rejected)
        with torch.no_grad():
            ref_c = seq_logprob(ref, instr, chosen)
            ref_r = seq_logprob(ref, instr, rejected)
        d_chosen = pol_c - ref_c        # Δ_chosen:相对参考模型,policy 给 chosen 多了多少
        d_rejected = pol_r - ref_r      # Δ_rejected
        # 【DPO-3】loss = −log σ( β·(Δ_chosen − Δ_rejected) )
        logit = args.beta * (d_chosen - d_rejected)
        loss = -F.logsigmoid(logit)
        if train:
            loss.backward()
        total_loss += loss.item()
        margin_sum += (d_chosen - d_rejected).item()
        rew_c_sum += args.beta * d_chosen.item()   # chosen 的隐式奖励(应往上走)
        rew_r_sum += args.beta * d_rejected.item() # rejected 的隐式奖励(应往下走)
        if (d_chosen - d_rejected).item() > 0:      # 隐式奖励把 chosen 排到 rejected 前面 = 判对
            correct += 1
    n = len(PREF)
    return total_loss / n, margin_sum / n, correct / n, rew_c_sum / n, rew_r_sum / n

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device} | 加载 SFT 模型作起点: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GPTConfig(**ckpt["config"])

    # 【DPO-1】policy 与 ref:都从 SFT 权重出发;ref 整段冻结、永不更新
    policy = GPT(cfg).to(device); policy.load_state_dict(ckpt["model"])
    ref = GPT(cfg).to(device);    ref.load_state_dict(ckpt["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    # 选手段:全量 DPO,或 LoRA 手段做 DPO
    if args.lora:
        for p in policy.parameters():
            p.requires_grad = False
        n_lora = inject_lora(policy, args.rank, args.alpha); policy.to(device)
        lr = max(args.lr, 2e-5)     # LoRA 参数少,适度放大学习率(仍远小于 SFT 的 LoRA lr)
        train_params = [p for p in policy.parameters() if p.requires_grad]
        tr = sum(p.numel() for p in train_params)
        print(f"手段=LoRA | 注入 {n_lora} 条旁路 r={args.rank} | 可训练 {tr:,}({100*tr/sum(p.numel() for p in policy.parameters()):.2f}%) | lr {lr}")
    else:
        lr = args.lr
        train_params = list(policy.parameters())
        print(f"手段=全量 DPO | 可训练 {sum(p.numel() for p in train_params):,} | lr {lr}")

    print(f"β={args.beta} | 偏好对 {len(PREF)} 条 | epochs {args.epochs}")

    # ---- DPO 前:基线诊断 + 采样 ----
    policy.eval()
    with torch.no_grad():
        l0, m0, a0, rc0, rr0 = dpo_pass(policy, ref, train=False)
    print(f"\n[DPO 前] loss {l0:.4f} | margin {m0:+.4f} | 准确率 {a0*100:.0f}% | 奖励 chosen {rc0:+.3f} / rejected {rr0:+.3f}")
    print("（起点 policy≡ref,两边奖励都≈0、margin≈0、loss≈ln2≈0.693 属正常）")
    show_samples(policy, "DPO 前 · 回答风格")

    # ---- DPO 训练 ----
    opt = torch.optim.AdamW(train_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    policy.train()
    print(f"\n开始 DPO …")
    for ep in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss, margin, acc, rc, rr = dpo_pass(policy, ref, train=True)   # 累计整批偏好对的梯度
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if ep % PRINT_EVERY == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d} | loss {loss:.4f} | margin {margin:+.3f} | 准确率 {acc*100:.0f}% | 奖励 chosen {rc:+.3f} / rejected {rr:+.3f}")

    # ---- DPO 后:诊断 + 采样 ----
    policy.eval()
    with torch.no_grad():
        l1, m1, a1, rc1, rr1 = dpo_pass(policy, ref, train=False)
    print(f"\n[DPO 后] loss {l1:.4f} | margin {m1:+.4f} | 准确率 {a1*100:.0f}% | 奖励 chosen {rc1:+.3f} / rejected {rr1:+.3f}")
    show_samples(policy, "DPO 后 · 回答风格")

    # ---- 存盘 ----
    if args.lora:
        sd = {k: v for k, v in policy.state_dict().items() if "lora_" in k}
        out = os.path.join(args.out_dir, "dpo_lora.pt")
        torch.save({"lora": sd, "rank": args.rank, "alpha": args.alpha,
                    "base_ckpt": args.ckpt, "beta": args.beta}, out)
    else:
        out = os.path.join(args.out_dir, "dpo.pt")
        torch.save({"model": policy.state_dict(), "config": cfg.__dict__,
                    "base_ckpt": args.ckpt, "beta": args.beta}, out)
    print(f"\n✅ DPO 完成,已存到 {out}")
    print(f"重点看硬指标:奖励 margin {m0:+.3f} → {m1:+.3f}、偏好准确率 {a0*100:.0f}% → {a1*100:.0f}%")
    print("（玩具级:采样文字变化可能很微妙,margin/准确率才是 DPO 起作用的可靠证据。）")

if __name__ == "__main__":
    main()
