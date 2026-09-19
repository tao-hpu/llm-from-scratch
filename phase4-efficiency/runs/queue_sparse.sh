#!/usr/bin/env bash
# 第 14 章:稀疏注意力 lm 四组
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
for br in full win cmp,sel cmp,sel,win; do
  tag=$(echo $br | tr ',' '-')
  $PY -u 12_sparse_attn.py --branches $br --json runs/ch14_lm_$tag.json > runs/ch14_lm_$tag.log 2>&1
done
echo ALL DONE
