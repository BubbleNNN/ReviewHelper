# 参与贡献

欢迎补充讲义、题目和问答卡，也欢迎纠错。所有内容都是 Markdown 和 Python 文件，改完提 Pull Request，CI 通过、合并后网页会自动更新。

## 最快的方式：改一处错误

每个网页底部都有“在 GitHub 上编辑此页”。点进去直接改，GitHub 会帮你 fork 并发起 PR。只想报告问题，就点“报告问题”开一个 issue。

## 本地准备

```bash
git clone <仓库地址> && cd ReviewHelper
uv sync                       # 装 numpy + pytest
uv run run.py --solution      # 参考答案应该全部 PASS
python3 build.py              # 生成 site/index.html
python3 -m http.server 8000 -d site   # 打开 http://localhost:8000 预览
```

## 加一篇讲义

1. 放到对应目录：`content/theory/rl/`、`content/theory/infra/`、`content/frameworks/<框架>/`。
2. 文件名只用小写字母、数字和连字符，例如 `kv-cache-offload.md`，它会成为网页地址 `#infra-kv-cache-offload`。
3. 开头写 front matter，`order` 决定在目录里的位置：

   ```markdown
   ---
   title: 标题
   summary: 一句话说明这篇讲什么
   tags: Infra, 推理
   order: 65
   ---
   ```

4. 正文支持：
   - `##` / `###` 标题（自动生成右侧提纲）
   - 公式 `$...$`、`$$...$$`（不要把 `$` 当货币符号用；写在 HTML 块里的公式要用 `\(...\)`）
   - GFM 表格、代码块
   - 提示框：`> [!NOTE]` 要点、`> [!TIP]` 技巧、`> [!WARN]` 易错、`> [!Q]` 面试追问、`> [!KEY]` 结论、`> [!SRC]` 源码对照
   - 简单流程图：`<div class="flow"><span>A</span><i>→</i><span class="f-rl">B<small>说明</small></span></div>`（`f-rl` / `f-infra` / `f-accent` 是三种底色）

## 加一道题

复制一个现成的题目目录（例如 `exercises/model/gae/`）改名，目录里四个文件：

| 文件 | 内容 |
|---|---|
| `problem.md` | front matter（`title`、`difficulty: 易/中/难`、`tags`、`order`、`source`）+ 题面；`<!-- solution -->` 之后是讲解，网页上点开答案才显示 |
| `template.py` | 函数签名 + 中文 docstring（写清楚形状和约定），函数体 `raise NotImplementedError` |
| `solution.py` | 参考答案 |
| `test_<slug>.py` | 测试，函数写成 `def test_xxx(m):`，`m` 是被测模块 |

测试的要求：

- 只用 numpy 和标准库，固定随机种子，每个用例 0.5 秒以内。
- 和**独立的参考**比：手算的数、暴力解、数学性质、有限差分。不要把参考答案的逻辑抄进测试。
- 允许多种正确答案时（比如拓扑序），检查性质而不是某个具体输出。

提交前跑：

```bash
uv run run.py <slug> --solution   # 必须全部 PASS
uv run run.py <slug>              # 模板必须是 TODO，不能是 FAIL / ERR
```

## 加问答卡

在 `content/qa/` 里找对应的卡组追加，或新建一个文件。每张卡是一个 `## 问题`，下面写答案（3–10 行为宜）。

## 加一个新领域

网站首页是领域选择页，每个领域（比如 RL × Infra）由若干**栏目**组成。栏目可以被多个领域共用：算法题栏目放进你的新领域里，就不用重写一遍。

1. 如果需要新栏目，先在 `content/sections.json` 里加一项：`id`（小写字母）、`group`（侧栏分组名）、`title`、`kind`（`doc` / `exercise` / `qa`）、`dir`（源目录）、`track`（`rl` / `infra` / `gen`，决定颜色）、`blurb`。然后建好对应目录，放内容。
2. 在 `content/domains.json` 里加一个领域：

   ```json
   {
     "id": "recsys",
     "title": "推荐系统",
     "tagline": "召回、排序与在线服务",
     "intro": "一两句话说明这个领域怎么复习。",
     "roles": ["推荐算法", "推荐工程"],
     "sections": ["recsys-theory", "algo", "recsys-qa"],
     "resources": [["名字", "https://...", "一句话说明"]]
   }
   ```

   `sections` 里的顺序不影响侧栏，侧栏按 `sections.json` 的顺序排；`resources` 是领域首页“外部资源”里的链接，可以不写。
3. `python3 build.py`，打开网页确认首页出现了新领域。

## 内容准确性

- 数字、默认值、论文结论尽量给出处（论文、源码路径、官方文档）。
- 没核实的写“待核实”，不要猜。面试复习材料，一个错的事实比缺一条更糟。
- 框架内容注明对照的版本或日期，例如“verl 主干，2026-11”。

## CI 会检查什么

每个 PR 会自动跑：参考答案全部通过、模板没有报错、pytest、网页能构建。合并到 `main` 后自动部署到 GitHub Pages。
