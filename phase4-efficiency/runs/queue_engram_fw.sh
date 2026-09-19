#!/usr/bin/env bash
# 第 15 章主实验:FineWeb-Edu token 级,四组,两条并行
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
($PY -u 13_engram.py --table 0 --json runs/ch15_fw_0.json > runs/ch15_fw_0.log 2>&1
 $PY -u 13_engram.py --table 262144 --orders 2 --json runs/ch15_fw_262144_2gram.json > runs/ch15_fw_262144_2gram.log 2>&1) &
($PY -u 13_engram.py --table 16384 --json runs/ch15_fw_16384.json > runs/ch15_fw_16384.log 2>&1
 $PY -u 13_engram.py --table 262144 --json runs/ch15_fw_262144.json > runs/ch15_fw_262144.log 2>&1) &
wait; echo ALL DONE
