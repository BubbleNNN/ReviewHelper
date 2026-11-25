---
title: KL 正则与熵
summary: KL 放 reward 还是放 loss、k1/k2/k3 三个估计量的值与梯度、verl 的 "+" 直通写法、自适应 KL 控制器，以及熵塌缩的成因与对策。
tags: RL, KL, 熵, verl
order: 50
---

## 为什么要 KL

RLHF 的标准目标是带 KL 约束的奖励最大化：

$$\max_\pi\ \mathbb E_{x,\,y\sim\pi}\big[R(x,y)\big] - \beta\, D_{\text{KL}}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$

参考模型 $\pi_{\text{ref}}$ 通常是 SFT 模型。KL 项的作用：防止 reward hacking（钻奖励模型的空子）、防止语言能力漂移（乱码、重复）、保留多样性。在可验证奖励（RLVR）的推理训练里，模型需要走得很远，DAPO、Dr.GRPO 都选择把 KL 去掉。

## 放在 reward 里还是 loss 里

<div class="cols">
<div><h4>KL in reward（经典 PPO-RLHF）</h4><p>每个 token 的奖励减去 \(\beta\log\frac{\pi_{\text{old}}}{\pi_{\text{ref}}}\)，再走 GAE。</p><p>KL 是回报的一部分，后续 token 的偏离会通过 return 影响前面 token 的优势；KL 值本身不求导。</p><p>verl：<code>algorithm.use_kl_in_reward=True</code>，<code>kl_penalty</code>，<code>kl_ctrl</code>。</p></div>
<div><h4>KL in loss（GRPO 风格）</h4><p>在 policy loss 上直接加 \(\beta\cdot\widehat{\text{KL}}(\pi_\theta,\pi_{\text{ref}})\)，逐 token 求导。</p><p>只惩罚当前 token 的局部偏离，梯度性质取决于用哪个估计量。</p><p>verl：<code>actor.use_kl_loss=True</code>，<code>kl_loss_coef=0.001</code>，<code>kl_loss_type=low_var_kl</code>。</p></div>
</div>

verl 的 reward 内 KL 很直接：

```python
def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio
```

只要 `use_kl_loss` 或 `use_kl_in_reward` 任一为真，trainer 就会创建 ref 模型并在每步计算 `ref_log_prob`。

> [!Q]
> **KL 放 reward 和放 loss 等价吗？**
>
> 不等价。放 reward 时，优化的是“回报里扣掉序列级 KL”的目标，策略梯度会考虑一个 token 对后续偏离的影响，KL 也会参与优势归一化；放 loss 时是逐 token 的局部正则，直接对估计量求梯度，和“KL 约束下最优策略”的目标并不完全一致。GRPO 论文给出的理由是不想让 $\hat A$ 的计算变复杂：它的优势是序列级、组内归一化的，而 KL 是 token 级的，放进 reward 就得重新设计优势。

## 三个估计量

记采样分布 $q=\pi_\theta$、目标分布 $p=\pi_{\text{ref}}$，样本 $x\sim q$，比值 $r = \frac{p(x)}{q(x)}$。要估计的是 $D_{\text{KL}}(q\,\|\,p) = \mathbb E_q[-\log r]$（Schulman, “Approximating KL Divergence”）：

| 名称 | 公式 | 期望（估计值） | 符号 | verl 名称 |
|---|---|---|---|---|
| k1 | $-\log r = \log q - \log p$ | 无偏 | 可正可负 | `kl` / `k1` |
| k2 | $\tfrac12(\log r)^2$ | 有偏（$p\approx q$ 时偏差很小） | 非负 | `mse` / `k2` |
| k3 | $(r-1) - \log r$ | 无偏（因 $\mathbb E_q[r]=1$） | 非负 | `low_var_kl` / `k3` |

k3 的直觉：$-\log r$ 加上一个期望为 0 的控制变量 $(r-1)$；又因为 $\log r \le r-1$，每个样本都非负，方差小。Schulman 博客的高斯例子里（$q=\mathcal N(0,1)$，$p=\mathcal N(0.1,1)$），k1 的标准差约为真值的 20 倍，k2、k3 约 1.4 倍（凭记忆引用，待核实）。

verl 对 k3 做了数值保护：`kl = clamp(ref_logprob - logprob, -20, 20)`，结果再 `clamp(-10, 10)`。

### 当 loss 用时看的是梯度

把样本当常数、只对 $\log q$ 求导：

- $\nabla k1 = \nabla\log q$，而 $\mathbb E_q[\nabla\log q]=0$。**k1 放进 loss，期望梯度为零**，只剩噪声。
- $\nabla k2 = (\log q - \log p)\nabla\log q$，期望正好是 $\nabla_\theta D_{\text{KL}}(q\,\|\,p)$。k2 的梯度是反向 KL 的无偏梯度。
- $\nabla k3 = (1 - \tfrac{p}{q})\nabla\log q$，期望是 $-\mathbb E_p[\nabla\log q] = \nabla_\theta D_{\text{KL}}(p\,\|\,q)$，是**正向 KL** 的梯度。仍然会把 $q$ 拉向 $p$，但不是目标里写的那个方向。

所以 verl 支持 `k1+`、`k3+` 这种带加号的写法：前向值用 k1/k3，反向梯度用 k2（straight-through）。

```python
backward_score = 0.5 * (logprob - ref_logprob).square()          # k2
return backward_score - backward_score.detach() + forward_score.detach()
```

数值上等于 `forward_score`，梯度等于 k2 的梯度。动手：exercises/model/kl_estimators。

> [!Q]
> **为什么 GRPO 默认用 k3（low_var_kl），不用 k1？**
>
> 作为“估计 KL 的数值”，k3 无偏、非负、方差小，适合做监控指标；作为 loss，k1 的期望梯度是 0，几乎没有正则效果。k3 的梯度对应正向 KL，方向仍然把策略拉回参考模型，实践中够用。追求“梯度严格对应反向 KL”就用 k2 或 `k3+`。

> [!WARN]
> 多个 mini-batch 时，loss 里的 KL 样本来自 $\pi_{\text{old}}$ 而不是当前 $\pi_\theta$，上面的期望推导只在第一个 mini-batch 严格成立。偏离不大时影响很小。

## 自适应 KL 控制器

Ziegler et al.（2019）的做法：KL 超过目标就加大 $\beta$，低于目标就减小。verl 的 `AdaptiveKLController.update`：

$$e = \operatorname{clip}\Big(\frac{\text{KL}_{\text{current}}}{\text{KL}_{\text{target}}} - 1,\ -0.2,\ 0.2\Big),\qquad \beta \leftarrow \beta\cdot\Big(1 + e\cdot\frac{n_{\text{steps}}}{\text{horizon}}\Big)$$

比例误差被截到 ±20%，`horizon` 控制调节速度。verl 默认 `algorithm.kl_ctrl.type=fixed`、`kl_coef=0.001`、`target_kl=0.1`、`horizon=10000`，自适应只对 reward 内 KL 生效。

## 熵与熵塌缩

token 级熵 $\mathcal H_t = -\sum_v \pi(v\mid s_t)\log\pi(v\mid s_t)$，verl 监控指标 `actor/entropy`。

**熵塌缩**：RLVR 训练初期熵快速下降，策略变得过度自信，之后探索不足、性能饱和。Cui et al.（2025, arXiv 2505.22617）给出两个观察：

- 经验关系 $R = -a\,e^{\mathcal H} + b$：性能提升几乎是用熵换来的，熵耗尽性能就到顶；
- 熵的变化由 **log-prob 与优势的协方差**驱动：高概率且优势为正的 token 被继续强化，会拉低熵；低概率且优势为正的 token 被提升则会增加熵。

<div class="flow"><span class="f-rl">高概率 token<small>A&gt;0 → 概率更高</small></span><i>→</i><span>分布更尖<small>熵下降</small></span><i>→</i><span>采样更单一</span><i>→</i><span class="f-accent">探索不足<small>性能饱和</small></span></div>

对策：

| 手段 | 原理 | verl |
|---|---|---|
| entropy bonus | loss 减去 $c_H\mathcal H$ | `actor.entropy_coeff`，默认 0；系数敏感，易熵爆炸 |
| clip-higher | 放宽上界，让低概率 token 能涨 | `clip_ratio_high=0.28` |
| Clip-Cov | 对协方差最大的少量 token 截断梯度 | `policy_loss.loss_mode=clip_cov` |
| KL-Cov | 对协方差最大的 top-k token 加 KL 惩罚 | `policy_loss.loss_mode=kl_cov` |
| 采样温度 | 训练 rollout 用 1.0 左右的温度 | `rollout.temperature`；算 logprob 时 logits 要除以同一温度 |
| 数据侧 | 去掉过简单的题、动态采样 | `filter_groups` |

> [!Q]
> **熵塌缩怎么在曲线上看出来？**
>
> `actor/entropy` 在前几十步断崖下降；同时 rollout 多样性下降（同一 prompt 的 n 条回答几乎一样，组内奖励全同，GRPO 优势为 0 的比例上升）；验证集 pass@k 不涨甚至下降而 pass@1 先涨后平。反过来如果熵持续上升、回答长度暴涨、出现乱码，就是熵爆炸，通常是 entropy_coeff 太大或训推不一致。

> [!SRC]
> `verl/trainer/ppo/core_algos.py`：`kl_penalty` / `kl_penalty_forward`（k1/abs/k2/k3/full 与 "+" 直通）、`compute_rewards`（reward 内 KL）、`AdaptiveKLController` / `FixedKLController`、`compute_entropy_loss`、`compute_policy_loss_clip_cov` / `compute_policy_loss_kl_cov`。

## 速记

- KL in reward：经典 RLHF，影响优势；KL in loss：GRPO，局部正则。
- k1 无偏高方差、梯度期望为 0；k2 有偏低方差、梯度 = 反向 KL 梯度；k3 无偏非负低方差、梯度 = 正向 KL 梯度。
- `k3+` = k3 的值 + k2 的梯度。
- 自适应 KL：$\beta \leftarrow \beta(1 + \operatorname{clip}(\text{KL}/\text{target}-1, \pm0.2)\cdot n/\text{horizon})$。
- 熵塌缩来自 logprob 与优势的正协方差；用 clip-higher、Clip-Cov/KL-Cov 缓解，慎用 entropy bonus。
