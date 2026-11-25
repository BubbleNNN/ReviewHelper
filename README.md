# ReviewHelper

**在线访问：<https://bubblennn.github.io/ReviewHelper/>** · 欢迎补充和纠错，见 [CONTRIBUTING.md](CONTRIBUTING.md)。

按领域组织的面试复习小站。打开网页先选领域；不同领域可以共用栏目（比如算法题）。目前的领域是 **RL × 大模型 Infra**：理论讲义、两个框架抓手（verl、DSpark）、带测试用例的代码题、翻卡式速问速答。所有内容都是 Markdown / Python 源文件，`build.py` 把它们打包成一个单文件网页。

## 快速开始

```bash
uv sync                      # 装 numpy + pytest（只需一次）
uv run run.py                # 跑全部题目，测试你写的 template.py
uv run run.py gae -v         # 只跑名字含 gae 的题，打印每个用例
uv run run.py --solution     # 用参考答案跑，确认测试本身没问题
uv run run.py --list         # 列出所有题
```

没有 uv 也可以：`pip install numpy`，然后把 `uv run` 换成 `python3`。也可以用 pytest：

```bash
uv run pytest                          # 测 template.py
RH_IMPL=solution uv run pytest         # 测参考答案
```

## 看网页

```bash
python3 build.py                       # 生成 site/index.html 和 site/artifact.html
python3 -m http.server 8000 -d site    # 浏览器打开 http://localhost:8000
```

`site/index.html` 是完整网页，本地直接用；`site/artifact.html` 是发布到 claude.ai Artifact 时用的页面片段。复习进度（已掌握、问答卡会/不会）只存在当前浏览器的 localStorage 里。

## 目录

```
content/
  domains.json           # 领域定义：每个领域引用哪些栏目、首页介绍、外部资源
  sections.json          # 栏目定义：标题、分组、岗位色（rl / infra / gen）、源目录
  site.json              # GitHub 仓库地址（网页里“在 GitHub 上编辑”用）
  theory/rl/*.md         # RL 理论
  theory/infra/*.md      # Infra 理论
  frameworks/verl/*.md   # verl：架构、训练循环、算法与损失、性能、异步与 Agent
  frameworks/dspark/*.md # DSpark：问题、草稿模型、调度、结果
  qa/*.md                # 问答卡，每个 "## 问题" 是一张卡
exercises/
  algo/<slug>/           # 算法题
  model/<slug>/          # 模型与 RL 实现（numpy）
  system/<slug>/         # 系统模拟（通信、KV cache、调度、投机采样…）
    problem.md           # 题面；<!-- solution --> 之后是讲解（网页上点开答案才显示）
    template.py          # 你要填的文件
    solution.py          # 参考答案
    test_<slug>.py       # 测试：def test_xxx(m)，m 是被测模块
run.py                   # 不依赖 pytest 的运行器
conftest.py              # pytest 的 m fixture
build.py                 # 打包网页
site/src/template.html   # 网页模板（样式 + 交互）
```

## 加内容

- **新讲义**：在对应目录放一个 `xxx.md`（文件名用小写字母和连字符），开头写 front matter：

  ```markdown
  ---
  title: 标题
  summary: 一句话摘要
  tags: RL, PPO
  order: 25
  ---
  ```

  正文支持 `$...$` / `$$...$$` 公式、GFM 表格、代码块，以及提示框：`> [!NOTE]` 要点、`> [!TIP]` 技巧、`> [!WARN]` 易错、`> [!Q]` 面试追问、`> [!KEY]` 结论、`> [!SRC]` 源码对照。

- **新题**：复制一个 `exercises/*/<slug>/` 目录改。测试只用 numpy 和标准库，函数签名写成 `def test_xxx(m):`。写完跑 `uv run run.py <slug> --solution` 确认全过，再跑 `uv run run.py <slug>` 确认模板显示 TODO。

- **新领域 / 新栏目**：栏目加在 `content/sections.json`，领域加在 `content/domains.json`（引用栏目 id），详见 [CONTRIBUTING.md](CONTRIBUTING.md#加一个新领域)。

## 在线访问与参与贡献

仓库推到 GitHub 后，`.github/workflows/pages.yml` 会在每次合并到 `main` 时自动测试、构建，并部署到 GitHub Pages；`ci.yml` 会在每个 Pull Request 上检查参考答案、模板和构建。

- 想补充或纠错：看 [CONTRIBUTING.md](CONTRIBUTING.md)。网页每页底部也有“在 GitHub 上编辑此页”和“报告问题”。
- 这两个链接用的仓库地址写在 `content/site.json` 的 `repo` 字段里，留空时网页上不显示。

## 内容说明

- verl 部分对照 2026-11 的主干源码（`0.10.0.dev`，V1 trainer）；配置默认值来自 `verl/trainer/config/_generated_ppo_trainer.yaml`。
- DSpark 部分依据论文 arXiv 2607.05147。论文后半部分（生产部署、附录）只读到摘要级别，正文里标了“待核实”。
- 其他标“待核实”的地方都是没能和一手资料对上的细节，引用前请再查。
