#!/usr/bin/env bash
# 第 12 章 Q&A:同一个 LLaMA 小模型,AdamW vs Muon
cd "$(dirname "$0")/../../phase3-moe"
PY=../.venv/bin/python
R=../phase4-efficiency/runs
$PY -u 10_llama.py --preset llama --json $R/ch12_adamw.json > $R/ch12_adamw.log 2>&1
$PY -u 10_llama.py --preset llama --optim muon --json $R/ch12_muon.json > $R/ch12_muon.log 2>&1
echo ALL DONE
