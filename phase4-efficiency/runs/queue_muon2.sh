#!/usr/bin/env bash
cd "$(dirname "$0")/../../phase3-moe"
../.venv/bin/python -u 10_llama.py --preset llama --optim muon --muon-lr 0.005 --json ../phase4-efficiency/runs/ch12_muon_lr005.json > ../phase4-efficiency/runs/ch12_muon_lr005.log 2>&1
