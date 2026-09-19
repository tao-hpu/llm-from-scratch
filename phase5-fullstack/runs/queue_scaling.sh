#!/usr/bin/env bash
# 第 20 章:scaling law 网格,串行跑(4090 上还常驻着别的进程,约 10GB)
# 5 个宽度 × 4 档 token 数 = 20 个拟合点;49M 的两个点不参加拟合,留作检验;另评第 6 章的 124M
cd "$(dirname "$0")/.."
PY=${PY:-python}
run(){ # L d h D [其他参数]
  f=runs/s_L$1_d$2_D$(($4/1000000))M.json
  [ -f $f ] && return
  $PY -u 18_scaling.py train --n-layer $1 --n-embd $2 --n-head $3 --tokens $4 "${@:5}" --json $f > ${f%.json}.log 2>&1
}
[ -f runs/s_124M_D10B.json ] || $PY -u 18_scaling.py evalckpt --ckpt ../phase1-124m/ckpt10b/latest.pt --tokens 10e9 \
  --json runs/s_124M_D10B.json > runs/s_124M_D10B.log 2>&1
for D in 50000000 100000000 200000000 400000000; do
  run 3 128 4 $D      # N ≈ 0.6M
  run 4 192 6 $D      # N ≈ 1.8M
  run 6 256 4 $D      # N ≈ 4.7M
  run 6 384 6 $D      # N ≈ 10.6M
  run 8 512 8 $D      # N ≈ 25M
done
run 10 640 10 100000000   # N ≈ 49M,检验点
run 10 640 10 200000000
# 注:L4_d192 训 4 亿 token 的点发散(第 2 层注意力分数涨到 1000 万量级),拟合时 --exclude 掉;
#     诊断与 qk-norm 对照见 runs/diverged/
# 延长 D:较大的三种尺寸各训 16 亿 token,把外推 124M(100 亿 token)的倍数从 25 倍降到约 6 倍;带诊断,盯住同样的发散
for s in "6 256 4" "6 384 6" "8 512 8"; do run $s 1600000000 --diag 1; done
# 124M 同结构(12 层 × 768)按本章配方训 4 亿 token:检验 N 方向外推(第 6 章是另一套 batch / warmup)
run 12 768 12 400000000 --diag 1
echo ALL DONE
