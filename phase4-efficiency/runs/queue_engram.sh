#!/usr/bin/env bash
# 第 15 章:N-gram 查表,五组(loss 对照;吞吐另测)
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
for cfg in "0 2,3" "4096 2,3" "32768 2,3" "32768 2" "1024 2,3"; do
  set -- $cfg
  tag=$1_$(echo $2 | tr ',' '-')
  $PY -u 13_engram.py --table $1 --orders $2 --json runs/ch15_$tag.json > runs/ch15_$tag.log 2>&1
done
echo ALL DONE
