#!/usr/bin/env bash
# 第 21 章:正式对照 + 消融。热身权重按配置自动命名(runs/ch21_warm_w<题数>_seed0_lr3e-05.pt),第一次跑时训出并存盘
cd "$(dirname "$0")/.."
PY=${PY:-python}
r(){ out=$1; shift; [ -f runs/$out.json ] || $PY -u 19_grpo.py "$@" --json runs/$out.json > runs/$out.log 2>&1; }
L0="--lr-sft 3e-5 --lr-dpo 3e-7 --lr-grpo 3e-6"                # 三条路的起始学习率;各起点再按 dev 另扫
r ch21_w200 $L0                                                # 热身 200 道:三条路(学习率扫描见 queue_grpo_lr.sh)
r ch21_w1000 --warm 1000 $L0                                   # 热身 1000 道:三条路
W="--warm 1000"
for G in 2 4 16; do r ch21_w1000_g$G --arms grpo --group $G --lr-grpo 3e-6 $W; done
r ch21_w1000_nokl --arms grpo --kl 0 --lr-grpo 3e-6 $W
# 热身 1000 道的起点上,三条路再各扫几个学习率(连同上面 ch21_w1000 的起始值),同样按 dev 挑;
# 最优值落在扫描区间边上就往外补点,直到最优值在区间内部(DPO 例外,见 queue_grpo_lr.sh 的说明)
for lr in 1e-6 3e-6 1e-5 1e-4; do r ch21_w1000_sft_$lr  --arms sft  --lr-sft  $lr $W; done
for lr in 1e-8 3e-8 1e-7 1e-6; do r ch21_w1000_dpo_$lr  --arms dpo  --lr-dpo  $lr $W; done
for lr in 1e-6 1e-5 3e-5; do r ch21_w1000_grpo_$lr --arms grpo --lr-grpo $lr $W; done
echo ALL DONE
