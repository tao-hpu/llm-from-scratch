#!/usr/bin/env bash
# 第 21 章:热身 200 道的起点上,每条路扫几个学习率,按 dev 题的贪心准确率挑(评测题不参与挑选)。全部在 RTX 4090 上跑。
# queue_grpo.sh 里的 ch21_w200(sft 3e-5 / dpo 3e-7 / grpo 3e-6)也算扫描的一个点。
# 最优值落在区间边上就往外补点;DPO 的 dev 准确率随学习率变小一路升高、逼近 warm 本身,没有区间内部的最优值。
cd "$(dirname "$0")/.."
PY=${PY:-python}
r(){ out=$1; shift; [ -f runs/$out.json ] || $PY -u 19_grpo.py "$@" --json runs/$out.json > runs/$out.log 2>&1; }
for lr in 1e-5 1e-4; do r ch21_lr_sft_$lr --arms sft --lr-sft $lr; done
for lr in 1e-8 3e-8 1e-7 1e-6 3e-6; do r ch21_lr_dpo_$lr --arms dpo --lr-dpo $lr; done
for lr in 3e-7 1e-6 1e-5 3e-5; do r ch21_lr_grpo_$lr --arms grpo --lr-grpo $lr; done
echo ALL DONE
