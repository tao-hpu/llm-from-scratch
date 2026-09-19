"""
里程碑 16(第 18 章):手写字节级 BPE —— 把第 6 章一直在调的 tiktoken 换成自己训的分词器
==============================================================================

第 1 章用的是字符级分词(65 个字符),第 6 章起一直直接调 tiktoken.get_encoding("gpt2")。
这一关自己从零训一个同类的分词器:

  【步骤 1】字节  文本先按 UTF-8 编成字节,初始词表就是 256 个字节,任何文本都编得出来,不会有"未知字符"
  【步骤 2】预切分  用 GPT-2 的正则把文本切成词、数字、标点、空白几类小段,合并只在段内发生,
                  所以 "dog." 和 "dog!" 里的 "dog" 是同一个 token,词表不会被 "dog." "dog!" "dog?" 这类变体占掉
  【步骤 3】数对  统计所有段里相邻两个 token 出现的次数
  【步骤 4】合并  出现最多的那一对合成一个新 token,记进合并表;回到步骤 3,直到词表够大
  编码:新文本按合并表的先后顺序,能合就合;解码:每个 token 查回它代表的字节,拼起来按 UTF-8 解开

思路来自 Sennrich et al.(arXiv 1508.07909)把 BPE 用到分词上;字节级 + 正则预切分是 GPT-2
(Radford et al. 2019,§2.2)的做法,正则原样取自 OpenAI 公开的 gpt-2 仓库 encoder.py。

训练语料:第 6 章那份 FineWeb-Edu val shard 解码回的英文原文(前 --train-mb MB),
测试语料取同一个 shard 靠后的一段,两段不重叠。

跑法(纯 CPU,默认参数约 1.5 分钟):
    python 16_bpe.py --json runs/ch18_bpe.json
    python 16_bpe.py --split space --json runs/ch18_bpe_space.json   # 对照:只按空格切,标点粘在词上
"""

import argparse
import collections
import json
import os
import time

import numpy as np
import regex
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--data", type=str, default=None,
               help="默认依次找 phase4-efficiency/data/ 与 phase1-124m/data/ 下的 edufineweb_val_000000.npy")
p.add_argument("--train-mb", type=float, default=8.0, help="训练语料多少 MB 英文原文")
p.add_argument("--test-mb", type=float, default=1.0)
p.add_argument("--vocab", type=int, default=16384, help="训到多大的词表(含 256 个字节)")
p.add_argument("--split", choices=["gpt2", "space"], default="gpt2",
               help="gpt2 = GPT-2 正则按字母 / 数字 / 符号分段;space = 只按空格切(对照)")
p.add_argument("--json", type=str, default="")
args = p.parse_args()

# GPT-2 的预切分正则(openai/gpt-2 src/encoder.py):
#   's 't 're 've 'm 'll 'd 这些缩写 | 可带一个前导空格的字母串 | 数字串 | 其他符号串 | 空白
GPT2_SPLIT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
# 对照:只按空格切,一个"词"连同它后面粘着的标点算一段
SPACE_SPLIT = regex.compile(r""" ?\S+|\s+""")


# ---------------------------------------------------------------------------
# 语料:把第 6 章的 token shard 解码回原文
# ---------------------------------------------------------------------------
def load_text():
    cands = [args.data] if args.data else [
        os.path.join(HERE, "..", "phase4-efficiency", "data", "edufineweb_val_000000.npy"),
        os.path.join(HERE, "..", "phase1-124m", "data", "edufineweb_val_000000.npy")]
    path = next((c for c in cands if c and os.path.exists(c)), None)
    assert path, "找不到 FineWeb-Edu val shard,先按第 6 章 prepare_fineweb.py 准备数据,或用 --data 指定"
    tok = np.load(path, mmap_mode="r")
    gpt2 = tiktoken.get_encoding("gpt2")
    # 按字节数估 token 数:这份语料 1 个 GPT-2 token 约 4.7 字节,多取一点再截
    need = int((args.train_mb + args.test_mb) * 1e6 / 4.0)
    # shard 里文档之间隔着 <|endoftext|>(id 50256),它是特殊 token 不是正文,换成空行
    text = gpt2.decode(tok[:need].tolist()).replace("<|endoftext|>", "\n\n")
    b = text.encode("utf-8")
    ntr = int(args.train_mb * 1e6)
    train = b[:ntr].decode("utf-8", errors="ignore")
    test = b[-int(args.test_mb * 1e6):].decode("utf-8", errors="ignore")
    return train, test


def chunks_of(text):
    return (GPT2_SPLIT if args.split == "gpt2" else SPACE_SPLIT).findall(text)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_bpe(text, vocab_size):
    """返回合并表 merges:[(a, b), ...],第 i 条合并产生 token 256 + i"""
    # 步骤 1、2:切段,同样的段只存一份并记次数(英文里 " the" 出现几万次,只算一次再乘次数)
    freq = collections.Counter(chunks_of(text))
    words = [list(w.encode("utf-8")) for w in freq]
    counts = list(freq.values())
    print(f"训练语料 {len(text.encode()):,} 字节 → {sum(counts):,} 段,去重后 {len(words):,} 段")

    # 步骤 3:数相邻对,同时记下每一对出现在哪些段里,合并时只改这些段
    pair_cnt = collections.Counter()
    where = collections.defaultdict(set)
    for i, w in enumerate(words):
        for a, b in zip(w, w[1:]):
            pair_cnt[a, b] += counts[i]
            where[a, b].add(i)

    merges, t0 = [], time.time()
    for new_id in range(256, vocab_size):
        if not pair_cnt:
            break
        # 步骤 4:出现最多的一对;次数相同取先出现的(Counter 按插入顺序),保证每次跑结果一样
        best = max(pair_cnt, key=pair_cnt.get)
        if pair_cnt[best] < 2:
            break
        merges.append(best)
        for i in list(where[best]):
            w, c = words[i], counts[i]
            for a, b in zip(w, w[1:]):              # 先把这个段的旧对子减掉
                pair_cnt[a, b] -= c
                if pair_cnt[a, b] <= 0:
                    del pair_cnt[a, b]
            j, nw = 0, []
            while j < len(w):                         # 把段里所有 best 换成 new_id
                if j + 1 < len(w) and (w[j], w[j + 1]) == best:
                    nw.append(new_id)
                    j += 2
                else:
                    nw.append(w[j])
                    j += 1
            words[i] = nw
            for a, b in zip(nw, nw[1:]):            # 再把新对子加回去
                pair_cnt[a, b] += c
                where[a, b].add(i)
        del where[best]
        if (new_id - 255) % 2000 == 0:
            print(f"  合并 {new_id-255:6d} 次 | 词表 {new_id+1:6d} | {time.time()-t0:.0f}s", flush=True)
    print(f"训练完:{len(merges)} 次合并,词表 {256 + len(merges)},{time.time()-t0:.0f}s")
    return merges


def token_bytes(merges):
    tb = [bytes([i]) for i in range(256)]
    for a, b in merges:
        tb.append(tb[a] + tb[b])
    return tb


# ---------------------------------------------------------------------------
# 编码 / 解码
# ---------------------------------------------------------------------------
def make_encoder(merges, limit=None):
    """只用前 limit 条合并(词表 = 256 + limit)。每个段内反复找"合并表里排得最靠前"的那一对来合"""
    rank = {pair: i for i, pair in enumerate(merges[:limit])}
    cache = {}

    def enc_chunk(s):
        if s in cache:
            return cache[s]
        w = list(s.encode("utf-8"))
        while len(w) >= 2:
            pairs = [(rank.get((a, b), 1 << 30), k) for k, (a, b) in enumerate(zip(w, w[1:]))]
            r, k = min(pairs)
            if r == 1 << 30:
                break
            w[k:k + 2] = [256 + r]
            # 同一对在段里可能出现多次,下一轮会接着合
        cache[s] = w
        return w

    def encode(text):
        out = []
        for s in chunks_of(text):
            out.extend(enc_chunk(s))
        return out
    return encode


def decode(ids, tb):
    return b"".join(tb[i] for i in ids).decode("utf-8", errors="replace")


def show(ids, tb):
    """每个 token 的文本(不完整的 UTF-8 片段用 <xx> 十六进制显示)"""
    out = []
    for i in ids:
        try:
            out.append(tb[i].decode("utf-8"))
        except UnicodeDecodeError:
            out.append("".join(f"<{x:02x}>" for x in tb[i]))
    return out


SENTENCES = {
    "en": "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "code": "for i in range(10):\n    total += i * i",
    "num": "In 2023, the population reached 8,045,311,447.",
    "zh": "光合作用把阳光转化为储存在葡萄糖里的化学能。",
}


def main():
    train, test = load_text()
    t0 = time.time()
    merges = train_bpe(train, args.vocab)
    train_s = time.time() - t0
    tb = token_bytes(merges)
    gpt2 = tiktoken.get_encoding("gpt2")
    test_bytes = len(test.encode("utf-8"))

    # 1) 往返:编码再解码必须一字不差
    full = make_encoder(merges)
    ids = full(test)
    assert decode(ids, tb) == test, "编码 → 解码 没有还原原文"
    print(f"往返检查通过:测试语料 {test_bytes:,} 字节 → {len(ids):,} token → 原样解码")

    # 2) 词表大小 vs 压缩率:同一段测试文本,每个 token 平均代表几个字节
    sizes = [s for s in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768) if s <= 256 + len(merges)]
    sweep = []
    for V in sizes:
        n = len(make_encoder(merges, V - 256)(test))
        sweep.append({"vocab": V, "tokens": n, "bytes_per_token": round(test_bytes / n, 3),
                      "emb_params_768": V * 768})
        print(f"  词表 {V:6d} | {n:9,} token | 每 token {test_bytes/n:.3f} 字节 | 嵌入表 {V*768/1e6:5.1f}M 参数(n_embd=768)")
    n_g = len(gpt2.encode(test))
    gpt2_row = {"vocab": gpt2.n_vocab, "tokens": n_g, "bytes_per_token": round(test_bytes / n_g, 3),
                "emb_params_768": gpt2.n_vocab * 768}
    print(f"  GPT-2(tiktoken) 词表 {gpt2.n_vocab} | {n_g:,} token | 每 token {test_bytes/n_g:.3f} 字节")

    # 3) 例句:不同合并次数下怎么切
    ks = [0, 10, 100, 1000, 4000, len(merges)]
    examples = {}
    for key, s in SENTENCES.items():
        rows = []
        for k in ks:
            e = make_encoder(merges, k)(s)
            rows.append({"merges": k, "tokens": show(e, tb)})
        g = gpt2.encode(s)
        rows.append({"merges": "gpt2", "tokens": [gpt2.decode_single_token_bytes(t).decode("utf-8", errors="replace")
                                                  if _ok(gpt2, t) else "".join(f"<{x:02x}>" for x in gpt2.decode_single_token_bytes(t))
                                                  for t in g]})
        examples[key] = {"text": s, "chars": len(s), "bytes": len(s.encode()), "rows": rows}
        print(f"  [{key}] {len(s)} 字符 / {len(s.encode())} 字节 → 本分词器 {len(rows[-2]['tokens'])} token,GPT-2 {len(g)} token")

    # 4) 最先学出来的合并,和每一段词表区间里的样例
    first = [{"rank": i, "a": show([a], tb)[0], "b": show([b], tb)[0], "tok": show([256 + i], tb)[0]}
             for i, (a, b) in enumerate(merges[:60])]
    bands = {}
    M = len(merges)
    for lo, hi in ((0, 100), (1000, 1100), (4000, 4100), (M - 100, M)):
        lo, hi = max(0, lo), min(hi, M)   # 词表小于 4,356 时后面的区间截短或跳过
        if lo < hi:
            bands[f"{lo}-{hi}"] = [show([256 + i], tb)[0] for i in range(lo, hi)]
    longest = sorted(range(len(merges)), key=lambda i: -len(tb[256 + i]))[:30]
    # 字母和标点粘在同一个 token 里的有多少(预切分就是为了不让词表花在这类变体上)
    LET, PUN = regex.compile(r"\p{L}"), regex.compile(r"[^\s\p{L}\p{N}]")
    toks = show(range(256, 256 + len(merges)), tb)
    mixed = [t for t in toks if LET.search(t) and PUN.search(t) and "<" not in t]
    print(f"字母 + 标点混在一个 token 里:{len(mixed)} 个(共 {len(toks)} 个合并出的 token)")

    out = {"config": vars(args), "train_bytes": len(train.encode()), "test_bytes": test_bytes,
           "train_seconds": round(train_s, 1), "n_merges": len(merges), "sweep": sweep, "gpt2": gpt2_row,
           "examples": examples, "first_merges": first, "bands": bands,
           "longest": [show([256 + i], tb)[0] for i in longest],
           "mixed_count": len(mixed), "mixed_examples": mixed[:60]}
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(out, open(args.json, "w"), ensure_ascii=False, indent=1)


def _ok(enc, t):
    try:
        enc.decode_single_token_bytes(t).decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


if __name__ == "__main__":
    main()
