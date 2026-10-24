"""
把各章的 .py 脚本生成 Jupyter Notebook(notebooks/*.ipynb)
=========================================================

为什么是"生成"而不是"手写":脚本是唯一正本,Notebook 只是它的另一种排版。
手写第二份代码迟早漂移;生成的话,改完脚本跑一遍本工具,Notebook 自动跟上。

切 cell 的依据是脚本里已有的分节注释,两种写法都认:
  # ---- 1. 标题 ----------------------------------            (单行)
  # ----------------------------------------------------------
  # 标题
  # ----------------------------------------------------------   (三行横幅,- 或 = 都行)
分节注释变成 markdown 标题,节与节之间的代码各成一个 code cell;
文件头的 docstring 变成第一格 markdown。

每本 Notebook 比脚本多一格"环境准备"code cell,也由本工具生成:
Colab 上 clone 仓库、缺包才装、进入章节目录,补上脚本靠 __file__ 定位数据的那个变量,
把命令行参数写进 sys.argv;需要预训练权重的章节,顺带从 HuggingFace 下载第 6 章训出的 124M 权重。

README 里的章节 ↔ Notebook 表,以及 web/ 各章节页顶部的「源码 / Colab」链接,也由本工具按 CHAPTERS
生成,都写在两行 HTML 注释标记之间(README: notebooks-table,页面: code-links)。

用法(不需要任何第三方库):
    python tools/build_notebooks.py           # 重新生成全部 Notebook 和 README 表
    python tools/build_notebooks.py --check   # 只比对,不一致则退出码 1(给 CI / pre-commit)
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "notebooks")
README = os.path.join(ROOT, "README.md")
README_MARK = ("<!-- notebooks-table:start -->", "<!-- notebooks-table:end -->")
REPO_URL = "https://github.com/tao-hpu/llm-from-scratch"
COLAB_BASE = REPO_URL.replace("https://github.com/", "https://colab.research.google.com/github/") + "/blob/main/notebooks/"
COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
HF_REPO = "tao-hpu/gpt2-124m-fineweb-edu-10b"
HF_FILE = "gpt2-124m-10b-val3.02.pt"
# 权重在仓库里的约定位置:各脚本的 --ckpt 默认值都指向这里
CKPT_REL = os.path.join("..", "phase1-124m", "ckpt10b", "latest.pt")

# import 名 → pip 包名。按脚本实际 import 决定装不装;装之前先 import 试一下,有就不装
PIP_BY_IMPORT = {"tiktoken": "tiktoken", "regex": "regex"}

# 章节表:脚本 → Notebook。只收不需要额外数据、或只需要公开权重的章节。
# 不收:04(预训练 1 天、要 N 卡)、13 / 15 / 16(要 FineWeb-Edu 验证片,仓库没发布)、
#       17(Triton kernel 要 N 卡)、18(训练要 N 卡)。
#   ch:          README / 网页的章号(与脚本编号不同)
#   needs:       README 表"需要什么"一栏
#   argv:        写进 sys.argv 的命令行参数(不含脚本名);"{CKPT}" 会换成权重绝对路径
#   needs_ckpt:  是否先下载 124M 权重
#   prereq:      先于本脚本执行的前置脚本及其参数(08 要先有 06 的 SFT 结果)
#   page:        web/ 下对应的章节页;页面顶部 .filename 角标后的代码 / Colab 链接也由本工具生成
CHAPTERS = [
    dict(ch=1, script="phase1-nanogpt/01_bigram.py", page="01_bigram_crossentropy_viz.html", needs="无,CPU 几分钟"),
    dict(ch=2, script="phase1-nanogpt/02_attention.py", page="02_attention_viz.html", needs="无"),
    dict(ch=3, script="phase1-nanogpt/03_transformer.py", page="03_transformer_viz.html", needs="无"),
    dict(ch=5, script="phase1-124m/05_sample.py", page="05_sampling_viz.html", needs="自动下载 124M 权重(约 500 MB)",
         needs_ckpt=True, argv=["--ckpt", "{CKPT}", "--prompt", "The history of ancient Rome", "--n", "3"]),
    dict(ch=8, script="phase2-sft-lora/06_sft.py", page="08_sft_viz.html", needs="同上",
         needs_ckpt=True, argv=["--ckpt", "{CKPT}", "--epochs", "30"]),
    dict(ch=9, script="phase2-sft-lora/07_lora.py", page="09_lora_viz.html", needs="同上",
         needs_ckpt=True, argv=["--ckpt", "{CKPT}"]),
    dict(ch=10, script="phase2-sft-lora/08_dpo.py", page="10_dpo_viz.html", needs="同上;没有 SFT 权重时先跑一遍 06",
         needs_ckpt=True, argv=[],
         prereq=dict(script="06_sft.py", argv=["--ckpt", "{CKPT}", "--epochs", "30"],
                     produces=os.path.join("ckpt_sft", "sft.pt"))),
    dict(ch=11, script="phase3-moe/09_moe.py", page="11_moe_viz.html", needs="无"),
    dict(ch=12, script="phase3-moe/10_llama.py", page="12_llama_viz.html", needs="无"),
    dict(ch=13, script="phase4-efficiency/11_linear_attn.py", page="13_linear_attention_viz.html", needs="无"),
    dict(ch=14, script="phase4-efficiency/12_sparse_attn.py", page="14_sparse_attention_viz.html", needs="无"),
    dict(ch=16, script="phase4-efficiency/14_mtp.py", page="16_speculative_decoding_viz.html", needs="无"),
    dict(ch=21, script="phase5-fullstack/19_grpo.py", page="21_rlvr_grpo_viz.html", needs="自动下载 124M 权重",
         needs_ckpt=True, argv=["--ckpt", "{CKPT}"]),
]

# 有配套脚本但没做成 Notebook 的章节页:只放 GitHub 源码链接
CODE_ONLY_PAGES = [
    dict(script="phase1-124m/04_gpt2_124m.py", page="07_pretraining_viz.html"),
    dict(script="phase4-efficiency/13_engram.py", page="15_ngram_memory_viz.html"),
    dict(script="phase4-efficiency/15_quant.py", page="17_quantization_viz.html"),
    dict(script="phase5-fullstack/16_bpe.py", page="18_bpe_viz.html"),
    dict(script="phase5-fullstack/17_flash_attn.py", page="19_flash_attention_viz.html"),
    dict(script="phase5-fullstack/18_scaling.py", page="20_scaling_law_viz.html"),
]
WEB_DIR = os.path.join(ROOT, "web")
WEB_MARK = ("<!-- code-links:start -->", "<!-- code-links:end -->")

RE_SINGLE = re.compile(r"^# -{4,} ?([^-\s].*?)\s*-{2,}\s*$")   # 单行:# ---- 标题 ----
RE_RULE = re.compile(r"^# [-=]{20,}\s*$")                       # 横幅的上下横线(- 或 =)


def nb_name_of(chapter):
    return os.path.splitext(os.path.basename(chapter["script"]))[0] + ".ipynb"


# ---------------------------------------------------------------------------
# 解析脚本
# ---------------------------------------------------------------------------
def split_docstring(src):
    """取出文件头的三引号 docstring,返回 (docstring 正文或 None, 剩余源码)。"""
    m = re.match(r'\s*(?:#[^\n]*\n)*\s*"""(.*?)"""\s*\n', src, re.S)
    if not m:
        return None, src
    return m.group(1).strip("\n"), src[m.end():]


def docstring_to_md(doc):
    """第一行当标题;正文逐行保留换行(行尾两个空格 = markdown 硬换行),缩进的命令行照常变代码块。"""
    lines = doc.split("\n")
    title = lines[0].strip()
    body = lines[1:]
    if body and re.fullmatch(r"=+\s*", body[0]):      # 标题下的 ==== 下划线
        body = body[1:]
    body = [ln.rstrip() + ("  " if ln.strip() else "") for ln in body]
    return "# " + title + "\n\n" + "\n".join(body).strip("\n")


def parse_sections(code):
    """按分节注释切块。返回 [(title or None, desc or None, code)]。
    横幅里第一行文字当标题,其余行当标题下的说明段。
    标题下面没有代码(紧跟下一个标题)也保留,只是不出 code cell。"""
    lines = code.split("\n")
    sections, cur_head, cur = [], (None, None), []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = RE_SINGLE.match(line)
        if m:
            sections.append((cur_head, cur))
            cur_head, cur = (m.group(1).strip(), None), []
            i += 1
            continue
        if RE_RULE.match(line):
            # 横幅:横线 / 若干 "# 文字" / 横线
            j, texts = i + 1, []
            while j < len(lines) and lines[j].startswith("#") and not RE_RULE.match(lines[j]):
                texts.append(lines[j].lstrip("#").strip())
                j += 1
            texts = [t for t in texts if t]
            if j < len(lines) and RE_RULE.match(lines[j]) and texts:
                sections.append((cur_head, cur))
                cur_head, cur = (texts[0], "\n".join(texts[1:]) or None), []
                i = j + 1
                continue
        cur.append(line)
        i += 1
    sections.append((cur_head, cur))
    out = []
    for (title, desc), c in sections:
        body = "\n".join(c).strip("\n")
        if title or body.strip():
            out.append((title, desc, body if body.strip() else ""))
    return out


# ---------------------------------------------------------------------------
# 生成 Notebook 开头两格 markdown + 环境准备 code cell
# ---------------------------------------------------------------------------
def setup_cell(chapter, src):
    chapter_dir, script = chapter["script"].split("/")
    pips = sorted((mod, pkg) for mod, pkg in PIP_BY_IMPORT.items()
                  if re.search(rf"^\s*(import|from)\s+{mod}\b", src, re.M))
    argv = chapter.get("argv", [])
    out = [
        "# ---- Notebook 环境准备 ----",
        "# 这一格只在 Colab / 新机器上做事:clone 仓库、装缺的包、进入章节目录。",
        "# 本地已 clone、在 notebooks/ 里打开的话,它只会切换目录。",
        "import os, sys, subprocess",
        f"REPO = {REPO_URL!r}",
        f"CHAPTER_DIR, SCRIPT = {chapter_dir!r}, {script!r}",
        "for root in ('.', '..', 'llm-from-scratch'):          # 当前目录 / notebooks 的上一级 / 已 clone 的副本",
        "    if os.path.exists(os.path.join(root, CHAPTER_DIR, SCRIPT)):",
        "        break",
        "else:",
        "    subprocess.run(['git', 'clone', '--depth', '1', REPO], check=True)",
        "    root = 'llm-from-scratch'",
        "os.chdir(os.path.join(root, CHAPTER_DIR))",
        "print('工作目录 =', os.getcwd())",
    ]
    if pips:
        out += [
            f"for mod, pkg in {pips!r}:                      # 脚本要用的第三方包,缺了才装",
            "    try:",
            "        __import__(mod)",
            "    except ImportError:",
            "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)",
        ]
    out += [
        "",
        "# 脚本用 __file__ 定位 data/ 等相对路径,Notebook 里没有这个变量,补上",
        "__file__ = os.path.abspath(SCRIPT)",
    ]
    if chapter.get("needs_ckpt"):
        out += [
            "",
            "# ---- 预训练权重:第 6 章在单卡 4090 上训出的 GPT-2 124M(FineWeb-Edu 10B token,val loss 3.02)----",
            "# 没有就从 HuggingFace 下载(约 500 MB),直接落到各脚本 --ckpt 默认指向的位置。",
            f"CKPT = os.path.abspath({CKPT_REL!r})",
            "if not os.path.exists(CKPT):",
            "    try:",
            "        from huggingface_hub import hf_hub_download",
            "    except ImportError:",
            "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], check=True)",
            "        from huggingface_hub import hf_hub_download",
            "    os.makedirs(os.path.dirname(CKPT), exist_ok=True)",
            f"    got = hf_hub_download({HF_REPO!r}, {HF_FILE!r}, local_dir=os.path.dirname(CKPT))",
            "    os.replace(got, CKPT)",
            "print('权重 =', CKPT)",
        ]
    pre = chapter.get("prereq")
    if pre:
        pre_argv = ", ".join("CKPT" if a == "{CKPT}" else repr(a) for a in pre["argv"])
        out += [
            "",
            f"# ---- 前置:本章接在 {pre['script']} 之后,没有它的输出就先跑一遍(几分钟)----",
            f"if not os.path.exists({pre['produces']!r}):",
            f"    subprocess.run([sys.executable, {pre['script']!r}, {pre_argv}], check=True)",
        ]
    argv_src = ", ".join("CKPT" if a == "{CKPT}" else repr(a) for a in argv)
    out += [
        "",
        "# 命令行参数:脚本用 argparse 读 sys.argv。想换参数,改这个列表再从头运行即可",
        f"sys.argv = [SCRIPT{', ' + argv_src if argv_src else ''}]",
    ]
    return "\n".join(out)


def header_md(chapter):
    rel = chapter["script"]
    return "\n".join([
        f"[![Open In Colab]({COLAB_BADGE})]({COLAB_BASE}{nb_name_of(chapter)})",
        "",
        f"> 本 Notebook 由 `tools/build_notebooks.py` 从 [`{rel}`]({REPO_URL}/blob/main/{rel}) 自动生成,"
        f"**代码以脚本为准,请勿直接改这里**;改了脚本重新生成即可。"
        f"交互式可视化见 [learn-llm.fim.ai](https://learn-llm.fim.ai),第 {chapter['ch']} 章。",
    ])


# ---------------------------------------------------------------------------
# 组装 .ipynb(纯 JSON,不依赖 nbformat;nbformat 4.4 不要求 cell id)
# ---------------------------------------------------------------------------
def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def build(chapter):
    src = open(os.path.join(ROOT, chapter["script"]), encoding="utf-8").read()
    doc, code = split_docstring(src)
    cells = [cell("markdown", header_md(chapter))]
    if doc:
        cells.append(cell("markdown", docstring_to_md(doc)))
    cells.append(cell("code", setup_cell(chapter, src)))
    for title, desc, body in parse_sections(code):
        if title:
            cells.append(cell("markdown", "### " + title + ("\n\n" + desc if desc else "")))
        if body:
            cells.append(cell("code", body))
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


# ---------------------------------------------------------------------------
# README 里的章节表
# ---------------------------------------------------------------------------
def readme_table():
    rows = ["| 章 | Notebook | Colab | 需要什么 |", "|:---:|---|---|---|"]
    for ch in CHAPTERS:
        name = nb_name_of(ch)
        rows.append(f"| {ch['ch']} | [`{name}`](notebooks/{name}) "
                    f"| [![Open In Colab]({COLAB_BADGE})]({COLAB_BASE}{name}) | {ch['needs']} |")
    return "\n".join(rows)


def replace_between(text, marks, content, what):
    """把 marks 两行标记之间的内容换成 content,保留标记所在行的缩进。"""
    start, end = marks
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        raise SystemExit(f"{what} 里找不到标记 {start} … {end}")
    line_start = text.rfind("\n", 0, i) + 1
    indent = text[line_start:i]
    if indent.strip():
        raise SystemExit(f"{what}:标记 {start} 必须独占一行")
    body = "".join(indent + ln + "\n" for ln in content.split("\n"))
    return text[:i + len(start)] + "\n" + body + indent + text[j:]


def readme_with_table(text):
    return replace_between(text, README_MARK, readme_table(), "README.md")


# ---------------------------------------------------------------------------
# 章节页顶部的代码 / Colab 链接(.filename 角标之后)
# ---------------------------------------------------------------------------
WEB_CSS_MARK = ("/* code-links-css:start */", "/* code-links-css:end */")
WEB_CSS = ".code-link{text-decoration:none;margin-left:6px;}\n.code-link:hover{border-color:var(--accent);color:var(--accent);}"


def page_links(entry, with_notebook):
    rel = entry["script"]
    links = [f'<a class="filename code-link" href="{REPO_URL}/blob/main/{rel}" target="_blank" rel="noopener">'
             f'📄 源码 {os.path.basename(rel)}</a>']
    if with_notebook:
        links.append(f'<a class="filename code-link" href="{COLAB_BASE}{nb_name_of(entry)}" target="_blank" rel="noopener">'
                     f'▶ 在 Colab 里跑这一章</a>')
    return "\n".join(links)


def page_targets():
    out = []
    for entry, with_nb in [(c, True) for c in CHAPTERS] + [(c, False) for c in CODE_ONLY_PAGES]:
        path = os.path.join(WEB_DIR, entry["page"])
        if not os.path.exists(path):
            raise SystemExit(f"CHAPTERS / CODE_ONLY_PAGES 里写的页面不存在:web/{entry['page']}")
        text = open(path, encoding="utf-8").read()
        text = replace_between(text, WEB_MARK, page_links(entry, with_nb), entry["page"])
        # 样式写进页面原有的 <style> 块(每页只许一个 <style>),第一次缺标记就在 </style> 前补上
        text = ensure_css_marks(text, entry["page"])
        text = replace_between(text, WEB_CSS_MARK, WEB_CSS, entry["page"])
        out.append((path, text))
    return out


# ---------------------------------------------------------------------------
# 首页卡片里的 Colab 入口(每张有 Notebook 的卡片,「开始 →」左边)
# ---------------------------------------------------------------------------
INDEX_MARK = ("<!-- colab-link:start -->", "<!-- colab-link:end -->")
INDEX_CSS = (".card-foot .colab{position:absolute;right:66px;top:9px;font-size:12px;color:var(--ink2);text-decoration:none;"
             "border:1px solid var(--border);border-radius:6px;padding:1px 7px;background:#fff;}\n"
             ".card-foot .colab:hover{color:var(--accent);border-color:var(--accent);}")


def ensure_css_marks(text, page):
    """页面的 <style> 块里没有样式标记就在 </style> 前补一对(每页只许一个 <style>)。"""
    if WEB_CSS_MARK[0] in text:
        return text
    m = re.search(r"^([ \t]*)</style>", text, re.M)
    if not m:
        raise SystemExit(f"web/{page} 里找不到 </style>")
    return text[:m.start()] + f"{m.group(1)}{WEB_CSS_MARK[0]}\n{m.group(1)}{WEB_CSS_MARK[1]}\n" + text[m.start():]


def index_target():
    path = os.path.join(WEB_DIR, "index.html")
    text = open(path, encoding="utf-8").read()
    for ch in CHAPTERS:
        card = re.search(rf'<article class="card" id="ch-{ch["ch"]:02d}">.*?</article>', text, re.S)
        if not card:
            raise SystemExit(f"index.html 里找不到第 {ch['ch']} 章的卡片 <article id=\"ch-{ch['ch']:02d}\">")
        body = card.group(0)
        if INDEX_MARK[0] not in body:
            # 第一次:标记放在「开始 →」那一行前面,同缩进
            m = re.search(r'^([ \t]*)<a class="go" ', body, re.M)
            if not m:
                raise SystemExit(f"index.html 第 {ch['ch']} 章卡片里找不到 <a class=\"go\">")
            body = body[:m.start()] + f"{m.group(1)}{INDEX_MARK[0]}\n{m.group(1)}{INDEX_MARK[1]}\n" + body[m.start():]
        link = (f'<a class="colab" href="{COLAB_BASE}{nb_name_of(ch)}" target="_blank" rel="noopener" '
                f'title="在 Colab 里跑这一章的代码">▶ Colab</a>')
        body = replace_between(body, INDEX_MARK, link, f"index.html 第 {ch['ch']} 章卡片")
        text = text[:card.start()] + body + text[card.end():]
    text = ensure_css_marks(text, "index.html")
    text = replace_between(text, WEB_CSS_MARK, INDEX_CSS, "index.html")
    return path, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只比对已生成的文件,不一致退出码 1")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = [(os.path.join(OUT_DIR, nb_name_of(ch)), build(ch)) for ch in CHAPTERS]
    readme_old = open(README, encoding="utf-8").read()
    targets.append((README, readme_with_table(readme_old)))
    targets += page_targets()
    targets.append(index_target())
    stale = []
    for path, text in targets:
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if args.check:
            if old != text:
                stale.append(path)
        else:
            if old != text:
                open(path, "w", encoding="utf-8").write(text)
            print(("更新 " if old != text else "不变 ") + os.path.relpath(path, ROOT))
    if args.check:
        if stale:
            print("以下文件与脚本不一致,请跑 python tools/build_notebooks.py 重新生成:")
            for p in stale:
                print("  ", os.path.relpath(p, ROOT))
            sys.exit(1)
        print(f"{len(CHAPTERS)} 本 Notebook、README 表、{len(CHAPTERS) + len(CODE_ONLY_PAGES)} 个章节页的代码链接均与脚本一致")


if __name__ == "__main__":
    main()
