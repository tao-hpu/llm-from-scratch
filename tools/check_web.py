"""
web/ 自检器 —— 把《可视化规范.md》里的硬规则变成可执行的检查
==============================================================

写它的原因：这些约定全靠人记，漏一条就是线上 bug。本次审计真实发现过的问题，
下面每一项都有对应检查：
  · 章节页少了 .home-link / .filename / .hp-bar / 深链 / 通关庆祝
  · 首页深链指向不存在的文件、超出目标页步数、或用错 hash 方案(#step- vs #part-)
  · 页面 data-key 在 glossary.html 里查不到(气泡点进去 404)
  · glossary 词条没有任何页面引用(孤儿词条，说明正文忘了加 .term)
  · glossary TOC 与词条对不上
  · 新增章节页忘了写进《可视化规范.md》的文件表
  · 违反"单文件零依赖离线可开"：引了外部 CDN / 字体 / 图片
  · 首页与子页的"共 N 步"对不上

跑法（不需要任何第三方库）：
    python tools/check_web.py
退出码 0 = 全过；1 = 有问题（可直接挂 CI / pre-commit）。
"""
import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
SPEC = os.path.join(ROOT, "可视化规范.md")

problems = []
notes = []


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def bad(msg):
    problems.append(msg)


def strip_scripts(s):
    s = re.sub(r"<script.*?</script>", "", s, flags=re.S)
    return re.sub(r"<style.*?</style>", "", s, flags=re.S)


pages = sorted(os.path.basename(p) for p in glob.glob(os.path.join(WEB, "*.html")))
# 章节页 = 形如 01_xxx_viz.html
chapters = [p for p in pages if re.match(r"^\d+_.*_viz\.html$", p)]
src = {p: read(os.path.join(WEB, p)) for p in pages}

# ---------------------------------------------------------------- 1. 页面铁律
def has_el(s, cls):
    """找的是"真有这个元素"，不是 CSS 里定义过这个类名 —— 否则删掉元素也检查不出来。"""
    return re.search(r'class="[^"]*\b%s\b[^"]*"' % re.escape(cls), s) is not None


for p in chapters:
    s = src[p]
    if not has_el(s, "home-link"):
        bad(f"{p}: 缺左上角 ← 返回首页(.home-link)")
    elif 'href="index.html"' not in s:
        bad(f"{p}: 有 .home-link 但没指回 index.html")
    if not has_el(s, "filename"):
        bad(f"{p}: 缺顶部 .filename 角标")
    if not has_el(s, "hp-bar"):
        bad(f"{p}: 缺常驻超参速查条(.hp-bar)")
    if f'>{p}<' not in s:
        bad(f"{p}: .filename 角标里的文件名和真实文件名不一致")
    # 深链：要么 #step-N（规范），要么 #part-N（04 的已知例外）
    if "'#step-'" not in s and "'#part-'" not in s:
        bad(f"{p}: 没有实现 hash 深链(#step-N)")
    if "celebrate" not in s and p != "04_gpt_machine_viz.html":
        bad(f"{p}: 缺通关庆祝 celebrate()")

for p in pages:
    s = src[p]
    # 单文件零依赖：不许引外部资源
    for m in re.finditer(r'(?:src|href)="(https?://[^"]+)"', s):
        url = m.group(1)
        if re.search(r"\.(js|css|woff2?|ttf|png|jpe?g|svg|gif)(\?|$)", url):
            bad(f"{p}: 引了外部资源，破坏离线可开：{url}")

# ---------------------------------------------------- 2. step 数 / panel 数自洽
step_count = {}
for p in chapters:
    s = src[p]
    tabs = len(re.findall(r'class="step-tab', s))
    # 第一个 panel 带 active，所以按 class="panel 前缀数（含 panel active）
    panels = len(re.findall(r'<section class="panel[ "]', s))
    step_count[p] = tabs
    if tabs == 0:
        bad(f"{p}: 一个 .step-tab 都没有")
    elif panels != tabs:
        bad(f"{p}: {tabs} 个 step-tab 但 {panels} 个 panel，对不上")

# --------------------------------------------------------- 3. 首页深链有效性
idx = src.get("index.html", "")
for m in re.finditer(r'href="([^":]+\.html)(#[^"]*)?"', idx):
    f, h = m.group(1), (m.group(2) or "")
    if f not in src:
        bad(f"index.html: 链到不存在的文件 {f}")
        continue
    tgt = src[f]
    ms = re.match(r"^#step-(\d+)$", h)
    mp = re.match(r"^#part-(\d+)$", h)
    if ms:
        n = int(ms.group(1))
        if "'#step-'" not in tgt:
            bad(f"index.html: {f}{h} —— 目标页用的不是 #step- 方案，深链会被忽略")
        elif n > step_count.get(f, 0):
            bad(f"index.html: {f}{h} 超出目标页步数({step_count.get(f)} 步)")
    elif mp:
        if "'#part-'" not in tgt:
            bad(f"index.html: {f}{h} —— 目标页不认 #part- 方案")
    elif h and not re.search(r'id="%s"' % re.escape(h[1:]), tgt):
        bad(f"index.html: {f}{h} —— 目标页没有这个锚点")

# 首页"共 N 步"要和子页 step 数一致
for m in re.finditer(r'href="(\d+_[^"#]*_viz\.html)#step-1"[^>]*>.*?</a>', idx, re.S):
    pass
for f, n in step_count.items():
    # 找首页里该章卡片附近的"共 N 步"
    for m in re.finditer(r'共\s*(\d+)\s*步', idx):
        pass
cards = re.split(r'<article|<div class="chapter', idx)
for c in cards:
    fm = re.search(r'href="(\d+_[^"#]*_viz\.html)#', c)
    nm = re.search(r'共\s*(\d+)\s*步', c)
    if fm and nm:
        f, n = fm.group(1), int(nm.group(1))
        if f in step_count and step_count[f] != n:
            bad(f'index.html: {f} 卡片写"共 {n} 步"，但页面实际 {step_count[f]} 步')

# --------------------------------------------------- 4. 名词气泡 ↔ glossary
g = src.get("glossary.html", "")
gids = set(re.findall(r'class="gentry"[^>]*id="([^"]+)"', g))
toc = set(re.findall(r'href="#([^"]+)"', g))
used = {}
for p in pages:
    for k in re.findall(r'data-key="([^"]+)"', src[p]):
        used.setdefault(k, set()).add(p)

for k, where in sorted(used.items()):
    if k not in gids:
        bad(f"glossary.html: 缺词条 #{k}(被 {', '.join(sorted(where))} 的气泡引用)")
for k in sorted(toc - gids):
    bad(f"glossary.html: TOC 链到 #{k} 但没有对应词条")
for k in sorted(gids - toc):
    bad(f"glossary.html: 词条 #{k} 没有加进顶部 TOC")
orphan = sorted(gids - set(used))
if orphan:
    notes.append("glossary 里没有任何页面引用的词条(正文可能忘了加 .term):\n    " + ", ".join(orphan))

# ------------------------------------------- 5. 规范文件表 vs 真实 web/ 目录
spec = read(SPEC)
for p in chapters + ["index.html", "glossary.html", "notes.html"]:
    if p not in spec:
        bad(f"可视化规范.md: 文件表里没有 {p}，新增页忘了登记")

# --------------------------------------------------------------------- 输出
print(f"检查 {len(pages)} 个页面(其中章节页 {len(chapters)} 个)\n")
for n in notes:
    print("· 提示:", n)
if problems:
    print(f"\n❌ {len(problems)} 个问题:")
    for x in problems:
        print("   -", x)
    sys.exit(1)
print("\n✅ 全部通过：铁律、深链、气泡词条、规范文件表都对得上。")
