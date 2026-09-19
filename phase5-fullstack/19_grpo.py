"""
里程碑 19(第 21 章):RLVR / GRPO —— 只给判分器、不给答案,把 124M 教会两位数加法
==============================================================================

Phase 2 的 SFT 和 DPO 都要"人给的东西":SFT 要标准答案,DPO 要一对好坏回答。
RLVR(Reinforcement Learning with Verifiable Rewards)只要一个能自动判对错的程序:

  【步骤 1】采样  同一道题让模型自己答 G 次(温度 1,每次答得可能不一样)
  【步骤 2】判分  判分器逐个检查:答对奖励 1,答错 0。判分器只会"对答案",不会"给答案"
  【步骤 3】组内比较  优势 A = (r − 组内均值) / 组内标准差。
                   同组比平均好的回答被加强,比平均差的被压低;G 个全对或全错时 A 全是 0,这道题不产生梯度
  【步骤 4】更新  loss = −A · log π(回答) + β · KL(π ‖ π_ref),和 SFT 同一个 log π,只是每条回答多乘了一个 A

组内均值当基线、不用 critic 网络,这是 GRPO(Shao et al.,DeepSeekMath,arXiv 2402.03300)§4.1 的做法。
这里每批采样只更新一次(严格同策略),所以 PPO 的比值裁剪不起作用,代码里略去。

实验设计(课程要求的"同一基座 / 同一评测 / 同样预算"):
  基座:第 6 章 10B token 训的 124M。它做两位数加法只有约 1% 对,采样 8 次几乎全错,RL 没有信号,
       所以先用 200 道带答案的题做一次热身 SFT,得到共同起点 warm。
  从 warm 出发三条路,每条都看同样的 4096 道训练题(和评测题不重叠):
    sft   4096 道题 + 标准答案,监督学习
    dpo   warm 对每道题采 8 次,判分器挑一对(对的当 chosen,错的当 rejected),离线 DPO
    grpo  每道题当场采 8 次,判分器打分,GRPO 更新
  评测:500 道没见过的题,贪心解码的准确率(pass@1),温度 1 采 8 次至少一次对的比例(pass@8),
       另按"有没有进位"分开统计。

学习率:每个起点上每条路各扫 3 到 5 个(runs/queue_grpo_lr.sh、runs/queue_grpo.sh),按 dev 题的贪心准确率挑,
       评测题不参与挑选。默认值 sft 3e-5、dpo 3e-7、grpo 3e-6 只是扫描的起始点;两个起点挑出的最优值不同
       (见第 21 章页面),换起点要重新挑。

跑法(CUDA / MPS / CPU 都能跑;RTX 4090 上三条路合计约 2 分钟):
    bash runs/queue_grpo_lr.sh && bash runs/queue_grpo.sh                  # 第 21 章页面的全部数字
    python 19_grpo.py --json runs/ch21_w200.json                           # 单跑一次:热身 200 道,三条路
    python 19_grpo.py --warm 1000 --lr-sft 3e-6 --lr-dpo 1e-7 --lr-grpo 1e-5 --json runs/ch21_w1000_best.json
    python 19_grpo.py --arms grpo --kl 0 --json runs/ch21_nokl.json          # 去掉 KL 项
    python 19_grpo.py --arms grpo --group 2 --json runs/ch21_g2.json         # 组更小
热身权重按配置自动存成 runs/ch21_warm_w<题数>_seed<种子>_lr<学习率>.pt,之后同配置的实验都从它出发。
"""

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=str, default=os.path.join(HERE, "..", "phase1-124m", "ckpt10b", "latest.pt"))
p.add_argument("--arms", type=str, default="sft,dpo,grpo")
p.add_argument("--warm", type=int, default=200, help="热身 SFT 用几道带答案的题")
p.add_argument("--prompts", type=int, default=4096, help="每条路看几道训练题")
p.add_argument("--batch", type=int, default=16, help="每步几道题")
p.add_argument("--group", type=int, default=8, help="每道题采样几次(GRPO 的 G,DPO 造数据也用它)")
p.add_argument("--lr-warm", type=float, default=3e-5, help="热身 SFT 的学习率,和 sft 那条路分开,换 --lr-sft 不影响起点")
p.add_argument("--warm-ckpt", type=str, default="",
               help="热身结果存在这里,文件已存在就直接加载,保证每次实验的起点完全相同。"
                    "默认按热身配置命名(runs/ch21_warm_w<题数>_seed<种子>_lr<学习率>.pt),旁边的 .json 记配置,加载时核对")
p.add_argument("--lr-sft", type=float, default=3e-5)
p.add_argument("--lr-dpo", type=float, default=3e-7)
p.add_argument("--lr-grpo", type=float, default=3e-6)
p.add_argument("--kl", type=float, default=0.04, help="GRPO 的 KL 系数 β(论文 §4.2 用 0.04)")
p.add_argument("--dpo-beta", type=float, default=0.1)
p.add_argument("--n-eval", type=int, default=500)
p.add_argument("--n-dev", type=int, default=200, help="挑学习率 / 画曲线用的 dev 题,和评测题、训练题都不重叠")
p.add_argument("--eval-every", type=int, default=32, help="每多少步在 dev 题上测一次贪心准确率")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--json", type=str, default="")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(args.seed)

# ---- 模型定义(与 05_sample.py 一致,去掉 KV cache)----
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
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

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
        self.ln_1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln_2, self.mlp = nn.LayerNorm(cfg.n_embd), MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, last_only=False):
        x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(idx.size(1), device=idx.device))
        for blk in self.transformer.h:
            x = blk(x)
        if last_only:                  # 生成时只要最后一个位置,省掉其余位置 × 50304 的 logits
            x = x[:, -1:]
        return self.lm_head(self.transformer.ln_f(x))

ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
base = GPT(GPTConfig(**ck["config"]))
base.load_state_dict(ck["model"])
base.to(device)
enc = tiktoken.get_encoding("gpt2")
NL = enc.encode("\n")[0]
VOCAB = 50257            # 50257 之后是补齐用的空行,采样时屏蔽

# ---------------------------------------------------------------------------
# 任务:两位数加法,数字逐位用空格隔开:"3 7 + 4 8 =" → " 8 5\n"。
# 不写成 "37 + 48 = 85":GPT-2 词表里 10…99、20…198 各是一个整体 token,模型只能死记 90×90 张表,
# 学不到按位相加(实测 4096 道题 SFT 后仍不到 10%)。逐位写之后每个数字一个 token,
# 题面固定 6 个 token,回答 2 到 3 个数字 + 换行
# ---------------------------------------------------------------------------
ALL = [(a, b) for a in range(10, 100) for b in range(10, 100)]
rng = random.Random(args.seed)
rng.shuffle(ALL)
EVAL = ALL[:args.n_eval]                          # 评测题:所有训练都不碰,只在最后报一次
DEV = ALL[args.n_eval:args.n_eval + args.n_dev]   # dev 题:画训练曲线、挑学习率
_o = args.n_eval + args.n_dev
WARM = ALL[_o:_o + args.warm]
POOL = ALL[_o + args.warm:]
TRAIN = [POOL[i % len(POOL)] for i in range(args.prompts)]   # 三条路看同一串题、同样的顺序

def spaced(n):
    return " ".join(str(n))

def prompt_ids(a, b):
    return enc.encode(f"{spaced(a)} + {spaced(b)} =")

def answer_ids(a, b):
    return enc.encode(f" {spaced(a + b)}\n")

P_LEN = len(prompt_ids(10, 10))
assert all(len(prompt_ids(a, b)) == P_LEN for a, b in ALL)

def carry(a, b):
    return (a % 10) + (b % 10) >= 10

def verify(a, b, resp_ids):
    """判分器:回答第一个换行之前的文本,去掉空格后是不是正确的和"""
    ids = resp_ids[:resp_ids.index(NL)] if NL in resp_ids else resp_ids
    return float(enc.decode(ids).replace(" ", "") == str(a + b))

MAX_NEW = 5

@torch.no_grad()
def generate(model, probs, n, temperature):
    """每道题生成 n 个回答。题面等长,不需要 padding。返回 [len(probs) * n] 个 token 列表"""
    rows = [prompt_ids(a, b) for a, b in probs for _ in range(n)]
    out = []
    for c in range(0, len(rows), 1024):              # 分批,评测时一次 4000 条也不占太多显存
        x = torch.tensor(rows[c:c + 1024], device=device)
        for _ in range(MAX_NEW):
            logits = model(x, last_only=True)[:, -1, :VOCAB].float()
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1)
            x = torch.cat([x, nxt], 1)
        for r in x[:, P_LEN:].tolist():              # 第一个换行之后的截掉
            out.append(r[:r.index(NL) + 1] if NL in r else r)
    return out

def seq_logprob(model, probs, resps):
    """每条 (题, 回答) 的回答部分 log π 之和,以及逐 token 的 log π 与掩码"""
    seqs = [prompt_ids(a, b) + r for (a, b), r in zip(probs, resps)]
    T = max(len(s) for s in seqs)
    x = torch.tensor([s + [NL] * (T - len(s)) for s in seqs], device=device)
    mask = torch.zeros_like(x, dtype=torch.float)
    for i, s in enumerate(seqs):
        mask[i, P_LEN:len(s)] = 1
    logp = F.log_softmax(model(x[:, :-1]).float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
    m = mask[:, 1:]
    return (logp * m).sum(1), logp, m

@torch.no_grad()
def evaluate(model, probs=None, k=8):
    probs = probs or EVAL
    model.eval()
    g = generate(model, probs, 1, 0)
    ok = [verify(a, b, r) for (a, b), r in zip(probs, g)]
    res = {"pass1": sum(ok) / len(ok),
           "pass1_carry": sum(o for o, pb in zip(ok, probs) if carry(*pb)) / max(1, sum(carry(*pb) for pb in probs)),
           "pass1_nocarry": sum(o for o, pb in zip(ok, probs) if not carry(*pb)) / max(1, sum(not carry(*pb) for pb in probs))}
    if k:
        s = generate(model, probs, k, 1.0)
        hit = [max(verify(a, b, s[i * k + j]) for j in range(k)) for i, (a, b) in enumerate(probs)]
        mean = sum(verify(a, b, s[i * k + j]) for i, (a, b) in enumerate(probs) for j in range(k)) / (len(probs) * k)
        res.update({f"pass{k}": sum(hit) / len(hit), "sample_acc": mean})
    res["examples"] = [f"{enc.decode(prompt_ids(a, b))}{enc.decode(r)!r}" for (a, b), r in list(zip(probs, g))[:12]]
    model.train()
    return {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in res.items()}

def quick(model):
    return round(evaluate(model, DEV, k=0)["pass1"], 4)

def sft_step(model, opt, probs):
    resps = [answer_ids(a, b) for a, b in probs]
    lp, _, m = seq_logprob(model, probs, resps)
    loss = -(lp.sum() / m.sum())            # 按 token 平均的交叉熵,只算回答部分(第 8 章的 loss mask)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# 三条路
# ---------------------------------------------------------------------------
def run_sft(model):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_sft, weight_decay=0.0)
    curve = []
    for s in range(len(TRAIN) // args.batch):
        loss = sft_step(model, opt, TRAIN[s * args.batch:(s + 1) * args.batch])
        if (s + 1) % args.eval_every == 0:
            curve.append({"step": s + 1, "loss": round(loss, 4), "acc": quick(model)})
            print(f"  sft  step {s+1:4d} | loss {loss:.4f} | 贪心 {curve[-1]['acc']:.3f}", flush=True)
    return curve, {}

def run_dpo(model, warm):
    # 造数据:warm 对每道题采 G 次,至少一对一错才能凑成一对
    pairs, t0 = [], time.time()
    for s in range(0, len(TRAIN), args.batch):
        probs = TRAIN[s:s + args.batch]
        rs = generate(warm, probs, args.group, 1.0)
        for i, (a, b) in enumerate(probs):
            g = rs[i * args.group:(i + 1) * args.group]
            good = [r for r in g if verify(a, b, r)]
            bad = [r for r in g if not verify(a, b, r)]
            if good and bad:
                pairs.append(((a, b), good[0], bad[0]))
    print(f"  dpo  造出 {len(pairs)} 对(共 {len(TRAIN)} 道题,{time.time()-t0:.0f}s)", flush=True)
    if not pairs:   # 起点太弱时 8 次全错,一对也凑不出来,DPO 没有数据可训
        print("  dpo  没有偏好对,跳过训练", flush=True)
        return [], {"n_pairs": 0, "pair_examples": []}
    ref = copy.deepcopy(warm).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_dpo, weight_decay=0.0)
    curve, steps = [], len(TRAIN) // args.batch
    for s in range(steps):   # 和另外两条路同样多的步数;对子数不够一步一批就循环用
        batch = [pairs[(s * args.batch + j) % len(pairs)] for j in range(args.batch)]
        probs = [q for q, _, _ in batch]
        pc, _, _ = seq_logprob(model, probs, [c for _, c, _ in batch])
        pr, _, _ = seq_logprob(model, probs, [r for _, _, r in batch])
        with torch.no_grad():
            rc, _, _ = seq_logprob(ref, probs, [c for _, c, _ in batch])
            rr, _, _ = seq_logprob(ref, probs, [r for _, _, r in batch])
        margin = args.dpo_beta * ((pc - rc) - (pr - rr))
        loss = -F.logsigmoid(margin).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (s + 1) % args.eval_every == 0:
            curve.append({"step": s + 1, "loss": round(loss.item(), 4), "acc": quick(model),
                          "pref_acc": round((margin > 0).float().mean().item(), 4)})
            print(f"  dpo  step {s+1:4d} | loss {loss.item():.4f} | 贪心 {curve[-1]['acc']:.3f}", flush=True)
    return curve, {"n_pairs": len(pairs), "pair_examples": [
        f"{enc.decode(prompt_ids(a, b))}  chosen{enc.decode(c)!r}  rejected{enc.decode(r)!r}" for (a, b), c, r in pairs[:8]]}

def run_grpo(model, warm):
    ref = copy.deepcopy(warm).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_grpo, weight_decay=0.0)
    G, curve, groups_log = args.group, [], []
    acc_r = acc_all0 = acc_all1 = acc_kl = 0.0
    for s in range(len(TRAIN) // args.batch):
        probs = TRAIN[s * args.batch:(s + 1) * args.batch]
        model.eval()
        resps = generate(model, probs, G, 1.0)                         # 【步骤 1】采样
        model.train()
        rep = [pb for pb in probs for _ in range(G)]
        r = torch.tensor([verify(a, b, x) for (a, b), x in zip(rep, resps)], device=device)   # 【步骤 2】判分
        rg = r.view(-1, G)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).view(-1)  # 【步骤 3】组内比较
        all1 = (rg.min(1).values == 1).float().mean().item()    # 全对:A 全 0
        all0 = (rg.max(1).values == 0).float().mean().item()    # 全错:A 全 0
        _, logp, m = seq_logprob(model, rep, resps)
        with torch.no_grad():
            _, ref_logp, _ = seq_logprob(ref, rep, resps)
        # KL 用论文 (4) 式的无偏估计:π_ref/π − log(π_ref/π) − 1,逐 token
        kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
        per_tok = -adv[:, None] * logp + args.kl * kl                   # 【步骤 4】更新
        loss = ((per_tok * m).sum(1) / m.sum(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        acc_r += r.mean().item(); acc_all0 += all0; acc_all1 += all1; acc_kl += ((kl * m).sum() / m.sum()).item()
        if s < 3 or (s + 1) % 64 == 0:
            i = 0
            groups_log.append({"step": s + 1, "prompt": enc.decode(prompt_ids(*probs[i])),
                               "answer": probs[i][0] + probs[i][1],
                               "samples": [enc.decode(x) for x in resps[:G]],
                               "reward": rg[i].tolist(), "adv": [round(v, 3) for v in adv[:G].tolist()]})
        if (s + 1) % args.eval_every == 0:
            n = args.eval_every
            curve.append({"step": s + 1, "reward": round(acc_r / n, 4), "all_wrong": round(acc_all0 / n, 4),
                          "all_right": round(acc_all1 / n, 4), "kl": round(acc_kl / n, 5), "acc": quick(model)})
            print(f"  grpo step {s+1:4d} | 采样平均奖励 {acc_r/n:.3f} | 全错组 {acc_all0/n:.2f} 全对组 {acc_all1/n:.2f} "
                  f"| KL {acc_kl/n:.4f} | 贪心 {curve[-1]['acc']:.3f}", flush=True)
            acc_r = acc_all0 = acc_all1 = acc_kl = 0.0
    return curve, {"groups": groups_log}


def main():
    out = {"config": vars(args), "device": device, "n_eval": len(EVAL),
           "eval_carry_frac": round(sum(carry(*pb) for pb in EVAL) / len(EVAL), 4)}
    t0 = time.time()
    print(f"device={device} | 评测 {len(EVAL)} 道(进位占 {out['eval_carry_frac']:.0%})| 热身 {len(WARM)} 道 | 每条路 {len(TRAIN)} 道")
    out["base"] = evaluate(base)
    print(f"基座 124M:贪心 {out['base']['pass1']:.3f} | pass@8 {out['base']['pass8']:.3f}")

    # 热身:200 道带答案的题,过 4 遍。训一次存盘,之后所有实验都从同一份权重出发
    warm = copy.deepcopy(base).train()
    warm_cfg = {"warm": args.warm, "seed": args.seed, "lr_warm": args.lr_warm, "batch": args.batch,
                "base": os.path.relpath(os.path.abspath(args.ckpt), HERE)}
    ckpt = args.warm_ckpt or os.path.join(HERE, "runs", f"ch21_warm_w{args.warm}_seed{args.seed}_lr{args.lr_warm:g}.pt")
    if os.path.exists(ckpt):
        meta = ckpt + ".json"
        if os.path.exists(meta):   # 同一个文件名被别的热身配置占了就报错,不悄悄换起点
            saved = json.load(open(meta))
            assert saved == warm_cfg, f"{ckpt} 是按 {saved} 训的,和这次的热身配置 {warm_cfg} 不一致"
        else:
            print(f"注意:{ckpt} 旁没有配置记录,无法核对热身配置")
        warm.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"加载热身权重 {os.path.relpath(ckpt, HERE)}")
    else:
        opt = torch.optim.AdamW(warm.parameters(), lr=args.lr_warm, weight_decay=0.0)
        for ep in range(4):
            for s in range(0, len(WARM), args.batch):
                sft_step(warm, opt, WARM[s:s + args.batch])
        os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
        torch.save(warm.state_dict(), ckpt)
        json.dump(warm_cfg, open(ckpt + ".json", "w"), ensure_ascii=False)
    warm.eval()
    out["warm"] = evaluate(warm)
    out["warm"]["dev"] = quick(warm)
    print(f"热身后 warm:贪心 {out['warm']['pass1']:.3f} | pass@8 {out['warm']['pass8']:.3f}")

    out["arms"] = {}
    for arm in args.arms.split(","):
        t1 = time.time()
        model = copy.deepcopy(warm).train()
        curve, extra = {"sft": lambda: run_sft(model), "dpo": lambda: run_dpo(model, warm),
                        "grpo": lambda: run_grpo(model, warm)}[arm]()
        final = evaluate(model)
        final["dev"] = quick(model)
        out["arms"][arm] = {"curve": curve, "final": final, "minutes": round((time.time() - t1) / 60, 1), **extra}
        print(f"[{arm}] 贪心 {final['pass1']:.3f}(进位 {final['pass1_carry']:.3f} / 不进位 {final['pass1_nocarry']:.3f})"
              f" | pass@8 {final['pass8']:.3f} | {out['arms'][arm]['minutes']} 分钟", flush=True)
        del model
    out["minutes"] = round((time.time() - t0) / 60, 1)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(out, open(args.json, "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
