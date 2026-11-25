---
title: 偏好学习与奖励设计
summary: RLHF 三阶段、Bradley-Terry 奖励模型、DPO 推导与它的变体、RLVR、ORM 与 PRM，以及 reward hacking 的常见形态和对策。
tags: RL, RLHF, DPO, 奖励模型
order: 60
---

## RLHF 三阶段

<div class="flow"><span>① SFT<small>示范数据监督微调，得到 \(\pi_{\text{SFT}}\)</small></span><i>→</i><span class="f-accent">② 奖励模型<small>人类偏好对 → Bradley-Terry</small></span><i>→</i><span class="f-rl">③ RL<small>PPO 最大化 RM 分数 − β·KL(π‖π_SFT)</small></span></div>

InstructGPT（Ouyang et al., 2022）确立了这个流程：标注员对同一 prompt 的多条回答排序，一个排序拆成两两比较训练 RM；RL 阶段用 PPO，奖励 = RM 分数 − 逐 token KL 惩罚。

## Bradley-Terry 奖励模型

假设人类偏好服从

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$$

最大似然得到 RM 损失：

$$\mathcal L_{\text{RM}} = -\mathbb E_{(x,y_w,y_l)}\big[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\big]$$

实现上 RM 是 SFT 模型去掉 LM head、换一个标量 head，读最后一个 token 的隐状态。只有**分数差**有意义，绝对值可以任意平移，所以训练后常做归一化（例如让 SFT 样本的平均分为 0）。

> [!WARN]
> RM 是真实偏好的不完美代理。Gao et al.（2022）的 RM 过优化实验显示：随着策略与初始策略的 KL 增大，代理 RM 分数单调上升，而“金标准”奖励先升后降。这是 RL 阶段必须有 KL 正则或早停的根本原因。

## DPO：把 RL 变成分类

### 第一步：KL 约束目标的闭式解

$$\max_\pi\ \mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)] - \beta\, D_{\text{KL}}(\pi\,\|\,\pi_{\text{ref}})$$

整理成一个 KL：

$$= -\beta\, D_{\text{KL}}\Big(\pi\ \Big\|\ \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\,e^{r(x,y)/\beta}\Big) + \beta\log Z(x)$$

KL 非负，最优解是

$$\pi^\star(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\Big(\frac{r(x,y)}{\beta}\Big),\qquad Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}$$

### 第二步：反解奖励

$$r(x,y) = \beta\log\frac{\pi^\star(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$$

$Z(x)$ 对所有 $y$ 求和，没法算；但它只依赖 $x$。

### 第三步：代入 Bradley-Terry，配分函数消掉

BT 只用到**同一个 $x$ 下两条回答的奖励差**，$\beta\log Z(x)$ 相减抵消：

$$\mathcal L_{\text{DPO}} = -\mathbb E\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]$$

隐式奖励 $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$。梯度：

$$\nabla_\theta\mathcal L_{\text{DPO}} = -\beta\,\mathbb E\Big[\sigma\big(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)\big)\big(\nabla\log\pi_\theta(y_w\mid x) - \nabla\log\pi_\theta(y_l\mid x)\big)\Big]$$

权重 $\sigma(\hat r_l - \hat r_w)$ 表示“模型现在排错得有多离谱”：排错越严重，更新越大。动手：exercises/model/dpo_loss。

> [!Q]
> **DPO 训练中 chosen 的 logprob 也在下降，正常吗？**
>
> 很常见。损失只约束 $\log\pi(y_w) - \log\pi(y_l)$ 的**差**变大，两者可以一起降，只要 rejected 降得更快；概率质量流向了数据之外的回答。风险是生成质量下降。缓解：在 chosen 上加 NLL/SFT 项、调 β、用 on-policy 数据做迭代 DPO。

> [!Q]
> **DPO 和 PPO 怎么选？**
>
> DPO 离线、简单、只需两个模型（policy + ref）的前向，没有 rollout，适合偏好数据充足的对齐阶段；但它只从固定数据学，没有探索，分布外泛化差。PPO / GRPO 在线采样，能从模型自己的错误里学，配合可验证奖励时上限更高，代价是系统复杂、要生成。推理能力的 RL 基本都是在线方法。

## DPO 的变体（一句话）

| 方法 | 改动 |
|---|---|
| IPO | 把 $-\log\sigma(h)$ 换成平方损失 $(h - \frac{1}{2\tau})^2$，偏好近乎确定时不会把 margin 推向无穷，缓解过拟合 |
| KTO | 不要成对数据，只要“好 / 坏”的单条标签，用前景理论（Kahneman-Tversky）的价值函数建模 |
| SimPO | 去掉参考模型，用长度归一化的平均 logprob $\frac{\beta}{\lvert y\rvert}\log\pi(y\mid x)$ 做隐式奖励，并加目标 margin $\gamma$ |
| ORPO | 不要参考模型，在 SFT 损失上加一项 odds ratio 惩罚，把 SFT 和偏好对齐合成一步 |

## RLVR：可验证奖励

推理类任务（数学、代码、逻辑）可以用程序判对错，直接给 0/1 奖励，不需要训练 RM。“RLVR”一词来自 Tülu 3（2024）；DeepSeek-R1 使用规则奖励：答案正确性 + 格式（例如思考过程放在指定标签里）。

在 verl 里自定义奖励：

```python
# custom_reward_function.path 指向这个文件，name 默认 compute_score
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred = extract_boxed_answer(solution_str)   # 解析失败时返回 None
    return 1.0 if pred is not None and is_equiv(pred, ground_truth) else 0.0
```

`reward_model.reward_manager` 默认 `naive`（逐条计算），`prime` 会并行验证；代码题可以接沙箱（配置里有 `sandbox_fusion`）。

> [!TIP]
> 规则奖励的工程要点：答案解析要严格且确定（只认最后一个 `\boxed{}`），否则模型会学会“多写几个答案”；超时、异常一律给 0；格式奖励的权重要小，避免模型只学格式；同一个 verifier 同时用于训练和评测，保证口径一致。

## ORM 与 PRM

| | ORM（结果奖励） | PRM（过程奖励） |
|---|---|---|
| 粒度 | 整条回答一个分 | 每个推理步骤一个分 |
| 标注 | 便宜，规则或单个标签 | 贵：人工标步骤（PRM800K）或用 MC rollout 自动估计（Math-Shepherd） |
| 用途 | RL 奖励、best-of-N | best-of-N 重排、树搜索、稠密奖励 |
| 风险 | 信号稀疏，信用分配难 | 步骤难定义；RL 中容易被 hack；需要随策略更新 |

Lightman et al.（2023, “Let's Verify Step by Step”）发现在 MATH 上做 best-of-N 重排时 PRM 明显优于 ORM。DeepSeek-R1 报告把 PRM 列为“没成功的尝试”：细粒度步骤难定义、中间步骤对错难判断、引入神经 RM 后会出现 reward hacking 且需要反复重训 RM。

> [!Q]
> **为什么 R1 这类推理 RL 偏向规则奖励而不用神经 RM / PRM？**
>
> 规则奖励无法被“讨好”，长时间训练也不会出现 RM 过优化；不需要维护和重训 RM，系统简单。代价是只适用于能自动判对错的任务，开放式写作、对话质量仍然需要 RM 或生成式 judge。

## reward hacking

常见形态：

- **长度**：RM 偏好长回答，策略就越写越长；
- **格式与表面特征**：堆砌 markdown、关键词，讨好 RM；
- **谄媚**：顺着用户的错误观点说；
- **钻 verifier 漏洞**：代码题针对测试用例写特判，数学题输出多个候选答案，利用解析器宽松匹配；
- **语言混杂**：推理 RL 中中英混写（R1-Zero 观察到的现象，R1 加了语言一致性奖励）。

对策：KL 正则与早停；RM 集成 / 不确定性惩罚；奖励裁剪与归一化；长度惩罚或长度无关的归一化；更严格的 verifier 和隐藏测试；定期用新策略的样本重训 RM；人工抽检 rollout。

> [!SRC]
> verl 的奖励入口：`custom_reward_function.path/name`（规则函数）、`reward_model.*`（RM worker，要求 `AutoModelForSequenceClassification`）、`reward_model.reward_manager`（naive / prime 等）。V1 trainer 中奖励由 `RewardLoopManager` 计算，`_step_once` 里对应 `reward` 计时段。

## 速记

- BT：$P(y_w\succ y_l) = \sigma(r_w - r_l)$，RM 损失 $-\log\sigma(r_w - r_l)$。
- DPO 三步：KL 约束最优解 → 反解奖励 → 代入 BT，$Z(x)$ 抵消。
- 隐式奖励 $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$；梯度权重 $\sigma(\hat r_l - \hat r_w)$。
- RLVR：规则判对错；ORM 稀疏但难 hack，PRM 稠密但贵且易 hack。
