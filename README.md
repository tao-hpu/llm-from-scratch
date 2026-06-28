# LLM from scratch ｜ 从 0 到 1 手搓大模型

> 用最少的代码、最白的中文，把一个语言模型从「预测下一个字符」一路搭到「复现 GPT-2 124M」。
> 不调包、不填 YAML，每一行都看得懂为什么。
>
> **English version → [jump to bottom](#-english).**

这不是又一个「开箱即用」的训练框架。它是一条**学习路线**:跟着 01 → 02 → 03 一步步把 Transformer 亲手搭出来,再用真实数据、真实 BPE 分词、真实优化器去复现 GPT-2。代码刻意写得短、注释刻意写得啰嗦,因为目标是**让你懂**,不是让你快。

灵感与结构来自 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 [build-nanogpt](https://github.com/karpathy/build-nanogpt)(见[致谢](#致谢))。本仓库的差异化是**逐行中文讲解**——给中文初学者一条不绕路的「先懂再快」之路。

---

## 交互式可视化

除了代码,每一关还配一套**交互式 HTML 可视化**——把抽象的张量、损失、注意力拆成可以亲手拨动的动画,**先看懂直觉,再读代码**。全部单文件、零依赖,**双击 [`web/index.html`](web/index.html) 即可离线打开**。

- [`web/index.html`](web/index.html) — 总入口,扁平列出所有章节
- [`web/01_bigram_crossentropy_viz.html`](web/01_bigram_crossentropy_viz.html) — 第 1 章:logits 摊平 & 交叉熵
- [`web/02_attention_viz.html`](web/02_attention_viz.html) — 第 2 章:自注意力,让每个位置"往回看"
- [`web/03_transformer_viz.html`](web/03_transformer_viz.html) — 第 3 章:完整 Transformer(多头 + FFN + 残差 + LayerNorm)
- [`web/04_gpt_machine_viz.html`](web/04_gpt_machine_viz.html) — 整机俯瞰:零件拼成 tiny GPT,装机 + 前向/采样全自动动画
- [`web/glossary.html`](web/glossary.html) — 名词表(术语字典,正文术语 hover 即弹气泡)
- [`web/notes.html`](web/notes.html) — 学习札记 / 彩蛋:正课之外的小故事(如 Transformer 前世今生:8 作者、翻译起源、家谱)

> 维护/新增可视化页面前,请先读 [`可视化规范.md`](可视化规范.md)——记录了配色、分步导航、按钮、名词气泡等全站约定。

---

## 为什么要手搓

- 只会用框架(LLaMA-Factory / TRL),你永远是「填配置的操作员」,模型一出问题就抓瞎。
- 全程手搓又是浪费时间重造生产基建。
- 正确顺序:**先手搓核心机制把原理吃透(Phase 1),再用生产工具做后训练(Phase 2)。**

---

## 仓库结构

```
phase1-nanogpt/      手搓阶段：字符级莎士比亚，从 bigram 到完整 Transformer
  01_bigram.py         里程碑①②：字符级 tokenizer + Bigram 基线（语言模型最朴素的样子）
  02_attention.py      里程碑③：从「对过去做加权平均」推导出 self-attention（QKV/因果掩码）
  03_transformer.py    里程碑④：拼出完整 GPT（多头注意力 + FFN + 残差 + 预归一化）
  导读.md              概念讲解 + 「训练脚本七步骨架」+ 超参旋钮手册
  data/                tinyshakespeare.txt（1MB，已含）

phase1-124m/         复现阶段：真实数据 + 真实 BPE，复现 GPT-2 124M
  prepare_fineweb.py   把 FineWeb-Edu 用 GPT-2 BPE 分词成 .npy shards
  prep_10b.sh          一键下载 + 分词 10B token 数据集
  04_gpt2_124m.py      GPT-2 124M 训练器（bf16 / Flash-Attn / 梯度累积 / 余弦退火）
  05_sample.py         加载 checkpoint 做推理、采样、续写

phase2-sft-lora/     【规划中 / WIP】SFT + LoRA 后训练
```

---

## 路线图

### Phase 1 — 手搓核心机制 ✅

- [x] 字符级 tokenizer + Bigram 基线,看懂 loss 为什么下降
- [x] 手写一遍 attention(QKV、因果掩码、多头)
- [x] 拼出完整 Transformer,在莎士比亚上训出能采样的模型
- [x] 真实数据 + 真实 BPE,复现 GPT-2 124M,理解 AdamW 每个超参

### Phase 2 — 用生产工具做后训练 🚧 规划中

1. **手写一次 LoRA**(在一个 Linear 上加 A、B 两个低秩矩阵,~30 行)——彻底懂 LoRA 是什么
2. **PEFT + TRL**:`SFTTrainer` / `DPOTrainer`,代码级控制
3. **LLaMA-Factory / Unsloth**:配置驱动、省显存,scale 和复现方便

> Phase 2 代码尚未提交,敬请期待。

---

## 训练成果（GPT-2 124M · FineWeb-Edu 10B）

在一张消费级 **RTX 4090（24G）** 上把 124M 模型在 **FineWeb-Edu 10B token** 上完整预训练了一遍,约 **1 天**跑完。

| 项 | 值 |
|---|---|
| 参数量 | 124M（GPT-2 small 配置） |
| 训练数据 | FineWeb-Edu，10B token（GPT-2 BPE，vocab 50257） |
| 硬件 / 耗时 | 单卡 RTX 4090，约 24 小时 |
| 训练步数 | 19073 步（cosine 退火，warmup 715） |
| 验证集 loss | 从随机初始化的 **≈10.9**（即 `ln(50257)` 的随机基线）降到 **3.02**（step 19072，val_loss=3.0211） |
| 推理 | CUDA / Apple MPS / CPU 通用；`05_sample.py` 内置 KV-cache,Mac MPS 上实测 **~2.5–2.8× 提速** |

**真实续写样本**（`05_sample.py`，prompt = `The history of ancient Rome`，`seed=1337` 可复现）:

```
The history of ancient Rome, particularly the last one, is quite fascinating.
The construction in the middle of the third century B.C. is known to have been
extremely complex. By the time of the construction of the great portico in the
city of Piazza dela, in about 400 B.C. the remains of an enormous fort, named
after Theodotus, were being built ... It was built of rock, with mortar being
mixed together with cement and clay.
```

句子通顺、扣题、用上了真实世界实体（portico、fort、mortar、B.C. 纪年），结构完整——相比小数据量训练出的「鬼打墙重复」是质的飞跃。

> ⚠️ **它在一本正经地瞎编史实**(Theodotus、"Piazza dela"、纪年全是拼凑)。这是 base 模型的本性:只学会「说得像」,没学会「说得对」。让它「说得对」是 Phase 2 SFT + 知识对齐要做的事。另外它**只会英文**(FineWeb-Edu 全英文语料),给中文 prompt 也只会蹦英文。

---

## 快速开始

### 0. 装依赖

```bash
pip install -r requirements.txt
```

> `torch` 请按你的平台到 [pytorch.org](https://pytorch.org) 选对应安装命令(CUDA / Mac MPS / CPU)。

### 1. Phase 1 玩具版(任何电脑都能跑,几分钟出结果)

```bash
cd phase1-nanogpt
python 01_bigram.py        # 看 loss 从 ~4.7 降到 ~2.5
python 02_attention.py     # 单头自注意力
python 03_transformer.py   # 完整 GPT，输出已有莎士比亚剧本的样子（ROMEO: / TYBALT:）
```

三个脚本自动选设备:**N 卡 → Apple MPS → CPU**。没有 GPU 也能跑,只是慢一点。

### 2. Phase 1 复现版 GPT-2 124M(全量预训练这一步需要 NVIDIA GPU)

```bash
cd phase1-124m

# (a) 准备数据：先小试 300M token（下载 1 个 parquet 即可）
#     wget 一个 parquet 到 data/ 后：
python prepare_fineweb.py --parquet "data/000_00000.parquet" --shards 3
#     或一键准备完整 10B token：
bash prep_10b.sh

# (b) 训练（单卡 4090 跑 10B 约 1 天，val loss 收敛到 ~3.0）
python 04_gpt2_124m.py --data_dir data10b --out_dir ckpt \
       --compile 1 --max_steps 19073 --warmup_steps 715

# (c) 玩一玩训出来的 base 模型（推理三平台通用：CUDA/MPS/CPU）
python 05_sample.py --ckpt ckpt/latest.pt --prompt "The history of Rome" --n 3
```

> ⚠️ 这是**预训练基座模型**,只会「续写」不会「答题」。给它 `Q: What is 2+2? A:` 它也只会接着编文本——「会回答问题」是 Phase 2 的 SFT 才教的。模型只用英文 FineWeb-Edu 训练,**不会中文,请用英文 prompt**。

---

## 硬件与平台

| 步骤 | CUDA(N卡) | Apple MPS | CPU |
|---|:---:|:---:|:---:|
| Phase 1 玩具版(01/02/03) | ✅ | ✅ | ✅ 慢 |
| GPT-2 124M **推理**(05) | ✅ | ✅ | ✅ 慢 |
| GPT-2 124M **全量预训练**(04) | ✅ 必需 | ❌ | ❌ |

预训练那一步用了 bf16 / `torch.compile` / CUDA-only 的优化,所以需要一张 NVIDIA GPU(单卡 24G 如 4090 足够)。**其余一切在 Mac 和纯 CPU 上都能跑**——所以 macOS、Windows、Linux 都能复现这条学习线,只是「自己从头预训练 124M」需要 N 卡。

---

## 预训练权重下载

为了保持仓库轻量,训练好的 checkpoint 不放进 git。可按上面的步骤自己训,或从下面下载(即将提供):

- 🤗 HuggingFace: _即将上传_
- 📦 GitHub Release: _即将上传_

---

## 致谢

整条路线深度参考 Andrej Karpathy 的 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列、[nanoGPT](https://github.com/karpathy/nanoGPT) 与 [build-nanogpt](https://github.com/karpathy/build-nanogpt)。本仓库在其基础上做了逐行中文重写与讲解,供中文初学者学习。数据集为 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)。

---

## 许可证

本项目以 [MIT License](LICENSE) 开源,可自由学习、修改、再发布。nanoGPT / build-nanogpt 同为 MIT,致谢见上。

---

## 🌍 English

**Build an LLM from scratch — with line-by-line Chinese explanations.**

A learning path, not a framework: hand-build a Transformer step by step (01 → 02 → 03), then reproduce GPT-2 124M with real data, real BPE, and a real optimizer. Code is kept short and over-commented on purpose — the goal is *understanding*, not speed.

Heavily based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [build-nanogpt](https://github.com/karpathy/build-nanogpt). **The added value here is the detailed Chinese walkthrough**, so the inline comments and `导读.md` are in Chinese. If there's interest in an English-commented version, open an issue.

**Quick start**

```bash
pip install -r requirements.txt

# Phase 1 — runs anywhere (CUDA → Apple MPS → CPU)
cd phase1-nanogpt && python 01_bigram.py && python 02_attention.py && python 03_transformer.py

# Phase 2 — reproduce GPT-2 124M (full pretraining needs an NVIDIA GPU)
cd ../phase1-124m
python prepare_fineweb.py --parquet "data/000_00000.parquet" --shards 3
python 04_gpt2_124m.py --data_dir data10b --out_dir ckpt --compile 1 --max_steps 19073 --warmup_steps 715
python 05_sample.py --ckpt ckpt/latest.pt --prompt "The history of Rome"
```

**Platform support**: the toy scripts (01–03) and inference (05) run on CUDA / Apple MPS / CPU. Only the 124M *full pretraining* step (04) requires an NVIDIA GPU (a single 24G card like a 4090 is enough), because it uses bf16, `torch.compile`, and CUDA-only optimizations. The model is trained on English-only FineWeb-Edu — it does **not** speak Chinese; use English prompts. It is a **base model**: it continues text, it does not answer questions (that's what Phase 2 SFT is for).
