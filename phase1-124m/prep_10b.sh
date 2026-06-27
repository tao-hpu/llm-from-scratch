#!/bin/bash
# 准备完整 10B token 数据集（下载剩余 parquet + 分词到 data10b/）
# 只用 CPU/网络/硬盘，不占 GPU，可与正在跑的 300M 训练并行。
# 分词输出到独立目录 data10b/，避免干扰 300M 训练正在读的 data/ 里的 .npy。
set -e
cd ~/llm-from-scratch/phase1-124m
BASE=https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT

echo "=== 1) 下载剩余 parquet（001~013，约 28GB）==="
for i in $(seq 1 13); do
  n=$(printf "%03d" $i)
  f="data/${n}_00000.parquet"
  echo "--- ${n} ---"
  wget -c -L --progress=dot:giga "${BASE}/${n}_00000.parquet" -O "$f"
done

echo "=== 2) 分词全部 14 个 parquet → data10b/（100 片 = 10B token）==="
python -u prepare_fineweb.py \
  --parquet "data/*.parquet" --shards 100 --out data10b

echo "=== 10B 数据准备完成 ==="
