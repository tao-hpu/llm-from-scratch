"""
重新生成分享卡片 web/og.png(1200×630)
====================================

为什么要有这个脚本:原来的 og.png 是一次性做好提交进来的,没有源文件。
它右下角写着"N 关",章节一多就悄悄过期(本次审计发现时,站上已有 11 章,图里还写着 10 关)。
所以把它做成可复现的脚本,**关数直接数 web/ 下的章节页**,以后不会再对不上。

跑法:
    python tools/make_og.py            # 重新生成 web/og.png
    python tools/make_og.py --check    # 只检查图里的关数是否还对得上(不写文件)

字体用 macOS 自带的 Hiragino Sans GB(中文)+ Menlo(等宽),
和站点 CSS 里 system-ui / var(--mono) 的实际渲染最接近。
"""
import argparse
import glob
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
OUT = os.path.join(WEB, "og.png")

# ---- 设计参数(量自原图,与 可视化规范.md 的配色一致)----
W, H = 1200, 630
BG      = (240, 238, 230)      # --bg  #F0EEE6
GRID    = (236, 233, 226)      # 竖向细网格线
INK     = (26, 26, 24)         # --ink #1A1A18
INK2    = (107, 107, 104)      # --ink2 #6B6B68
ACCENT  = (217, 119, 87)       # --accent #D97757
PAD_L   = 72

CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"


def font(size, bold=False, mono=False):
    if mono:
        return ImageFont.truetype(MONO, size, index=1 if bold else 0)
    return ImageFont.truetype(CJK, size, index=2 if bold else 0)   # 2 = W6(粗)


def count_chapters():
    """关数 = web/ 下形如 01_xxx_viz.html 的章节页个数。"""
    return len([p for p in glob.glob(os.path.join(WEB, "*_viz.html"))
                if re.match(r"^\d+_", os.path.basename(p))])


def draw_mixed(d, xy, parts, f):
    """一行里混排不同颜色的文字,返回结束时的 x。"""
    x, y = xy
    for text, color in parts:
        d.text((x, y), text, font=f, fill=color)
        x += d.textlength(text, font=f)
    return x


def build(n_chapters):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    # 三条竖向细网格线(把画面切成四栏,原图就有)
    for x in (300, 600, 900):
        d.rectangle([x - 1, 0, x, H], fill=GRID)

    # 左上角橙色短杠
    d.rectangle([PAD_L, 84, PAD_L + 131, 91], fill=ACCENT)

    # 眉标
    d.text((PAD_L, 122), "交互式教程  ·  Python 真实现 + 可视化", font=font(25), fill=INK2)

    # 主标题:前四字黑、后五字橙
    f_title = font(88, bold=True)
    draw_mixed(d, (PAD_L, 208), [("从零手写", INK), ("大语言模型", ACCENT)], f_title)

    # 副标题两行
    d.text((PAD_L, 366), "每一关 = 真实现的 Python 代码 + 可亲手拨动的可视化",
           font=font(34), fill=INK)
    d.text((PAD_L, 428), "先建直觉,再落到代码:bigram 一路搭到 GPT、对齐与现代架构。",
           font=font(25), fill=INK2)

    # 底部:左边站点,右边关数
    d.rectangle([PAD_L, 545, PAD_L + 15, 560], fill=ACCENT)
    d.text((PAD_L + 28, 538), "learn-llm.fim.ai", font=font(27, mono=True), fill=ACCENT)

    right = f"{n_chapters} 关 · 持续更新"
    f_r = font(25)
    d.text((W - 72 - d.textlength(right, font=f_r), 540), right, font=f_r, fill=INK2)
    return im


def current_label():
    """粗略读出现有 og.png 是按几关做的(靠文件旁边的记号文件,没有就返回 None)。"""
    mark = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".og_chapters")
    if os.path.exists(mark):
        return int(open(mark).read().strip())
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查关数是否对得上,不写文件")
    a = ap.parse_args()

    n = count_chapters()
    was = current_label()
    if a.check:
        if was is None:
            print(f"⚠️  没有记号文件,无法确认 og.png 是按几关做的(当前 {n} 关)。跑一次不带 --check 重新生成即可。")
            sys.exit(1)
        if was != n:
            print(f"❌ og.png 是按 {was} 关做的,现在有 {n} 关 —— 需要重新生成:python tools/make_og.py")
            sys.exit(1)
        print(f"✅ og.png 与章节数一致({n} 关)")
        sys.exit(0)

    build(n).save(OUT, optimize=True)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".og_chapters"), "w").write(str(n))
    print(f"已生成 {OUT}(按 {n} 关渲染,{os.path.getsize(OUT)/1024:.0f} KB)")
