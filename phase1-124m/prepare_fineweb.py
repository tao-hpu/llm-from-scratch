"""
数据预处理：把 FineWeb-Edu 用 GPT-2 真 BPE 分词成 shards
======================================================

这是 build-nanogpt 相比我们 phase1 玩具版多出来的【第一样东西】：真实数据 + 真实分词。
  phase1：1MB 莎士比亚 + 字符级 tokenizer（vocab=65）
  这里  ：FineWeb-Edu（高质量教育网页）+ GPT-2 BPE（tiktoken，vocab=50257，子词级）

【为什么不用 datasets/hf_hub_download】这台服务器上 streaming 首次拉取会网络卡死，
hf_hub_download 又因镜像重定向校验失败。所以最稳的办法：parquet 文件用 wget -c 提前下到本地，
这里只负责读本地 parquet（pyarrow 流式）+ 批量分词。sample/10BT 共 14 个 parquet，
每个约 7 亿 token，1 个就够有界运行。

输出：data/edufineweb_val_000000.npy（第 0 片留作验证）
      data/edufineweb_train_000001.npy …（其余作训练），每片 1 亿 token，uint16

用法：python prepare_fineweb.py --parquet data/000_00000.parquet --shards 3
"""
import os
import glob
import argparse
import numpy as np
import tiktoken
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", type=str, default="data/000_00000.parquet", help="本地 parquet 路径（可用通配）")
parser.add_argument("--shards", type=int, default=3, help="一共生成多少片（含 1 片验证）")
parser.add_argument("--shard_size", type=int, default=10**8, help="每片 token 数，默认 1 亿")
parser.add_argument("--out", type=str, default="data")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

PARQUETS = sorted(glob.glob(args.parquet))
assert PARQUETS, f"找不到 parquet 文件：{args.parquet}"

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]   # 50256
print(f"tiktoken gpt2: vocab={enc.n_vocab}, eot={eot}", flush=True)

buf = np.empty((args.shard_size,), dtype=np.uint16)
count, shard_idx, docs = 0, 0, 0

def write_shard(idx, data):
    split = "val" if idx == 0 else "train"
    path = os.path.join(args.out, f"edufineweb_{split}_{idx:06d}.npy")
    np.save(path, data)
    print(f"  ✓ {path}  ({len(data):,} tokens)", flush=True)

def add_tokens(toks):
    """把一篇文档的 token 填进当前片缓冲区，满了就写出、开新片。返回 False 表示已达目标片数。"""
    global count, shard_idx
    i = 0
    while i < len(toks):
        space = args.shard_size - count
        take = min(space, len(toks) - i)
        buf[count:count + take] = toks[i:i + take]
        count += take
        i += take
        if count == args.shard_size:          # 当前片填满
            write_shard(shard_idx, buf)
            shard_idx += 1
            count = 0
            if shard_idx >= args.shards:
                return False
    return True

done = False
for local in PARQUETS:
    if done:
        break
    print(f"分词 {local} …", flush=True)
    pf = pq.ParquetFile(local)
    for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
        texts = batch.column("text").to_pylist()
        # encode_ordinary_batch 内部多线程，比逐条 encode 快
        for tokens in enc.encode_ordinary_batch(texts):
            docs += 1
            if not add_tokens([eot] + tokens):
                done = True
                break
        if done:
            break
        if docs % 20000 == 0:
            prog = shard_idx + count / args.shard_size
            print(f"  …{docs:,} 篇文档，进度 {prog:.2f}/{args.shards} 片", flush=True)

# 说明：只产出"填满 shard_size"的完整片;循环结束时 buf 里剩的不足一片的尾部 token 会被丢弃
# （val 片 idx 0 总是先填满,不受影响;丢的是训练尾巴）。nanoGPT 原版会把残片也写出,这里从简。
print(f"\n完成：{shard_idx} 片（目标 {args.shards}），共 {docs:,} 篇文档。", flush=True)
