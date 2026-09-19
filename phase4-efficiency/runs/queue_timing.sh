#!/usr/bin/env bash
# 需要安静环境的测量:等其他训练队列都结束再跑(墙钟 / 吞吐数字才不被抢占)
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
while pgrep -f "queue_ch13|queue_moe|queue_engram|queue_sparse|queue_muon|15_quant.py" >/dev/null; do sleep 60; done
# 第 15 章吞吐:同样 400 步,表在 GPU / 表在 CPU / 不挂表
for cfg in "0 same" "32768 same" "32768 cpu"; do
  set -- $cfg
  $PY -u 13_engram.py --table $1 --table-device $2 --max-iters 400 --eval-iters 1 --json runs/ch15_speed_$1_$2.json > runs/ch15_speed_$1_$2.log 2>&1
done
# 第 16 章:主模型 + MTP + 草稿模型 + 投机解码测量;再训一个不带 MTP loss 的对照
$PY -u 14_mtp.py --json runs/ch16_mtp.json > runs/ch16_mtp.log 2>&1
$PY -u 14_mtp.py --mtp 0 --no-spec --json runs/ch16_nomtp.json > runs/ch16_nomtp.log 2>&1
echo ALL DONE
