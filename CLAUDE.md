# 项目说明(给 Claude 的工作约定)

本项目 = 从零手写大语言模型的**教学项目**:每一关有 Python 代码(真正实现)+ 交互式可视化(先建直觉,页面在 learn-llm 仓库)。

## 网页在 learn-llm 仓库

交互式可视化页面已迁到 [tao-hpu/learn-llm](https://github.com/tao-hpu/learn-llm)(React,线上 https://learn-llm.fim.ai),本仓库不再放 HTML。改页面、加章节、配色与组件约定都去那个仓库做。

- 本仓库只放 Python 代码(按 `phaseX-*/` 分目录)与 `notebooks/`。
- **某章新增或改动了脚本 / Notebook,要同步改 learn-llm 的 `src/catalog.json` 里该章的 `codeLinks` / `source` / `colab`**:`tools/build_notebooks.py` 不再往网页注入链接。

## 教学风格

像第 1 章那样**真正一步步推进**:5 步左右一关,每步一个可亲手拨动的交互 + `↳ 下一步` 钩子,先给直觉再落到代码。一关讲清一条主线,别把页面堆成名词解释。

## notebooks/ 是生成物,不手改

`notebooks/*.ipynb` 由 `tools/build_notebooks.py` 从各章 `.py` 生成(分节注释 → markdown 标题,节间代码 → code cell)。
改脚本后跑 `python tools/build_notebooks.py` 重新生成;`--check` 校验一致。新章节要进 Notebook,在该工具的 `CHAPTERS` 表加一行。
