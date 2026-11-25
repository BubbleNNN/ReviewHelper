---
title: TRPO 与 PPO
summary: 从性能差引理和 trust region 讲到 PPO-clip 的逐情况分析、dual-clip、clip-higher、value clip、GAE 和 verl 里的训练结构。
tags: RL, PPO, TRPO, verl
order: 30
---

## 为什么需要 trust region

**性能差引理**（Kakade & Langford, 2002）：

$$J(\pi') - J(\pi) = \mathbb E_{\tau\sim\pi'}\Big[\sum_t \gamma^t A^{\pi}(s_t,a_t)\Big]$$

右边的状态来自新策略 $\pi'$，没法直接优化。把状态分布换成旧策略的，得到代理目标

$$L_\pi(\pi') = J(\pi) + \mathbb E_{s\sim d^\pi,\, a\sim\pi}\Big[\frac{\pi'(a\mid s)}{\pi(a\mid s)}A^\pi(s,a)\Big]$$

它在 $\pi'=\pi$ 处与真实目标一阶相同。TRPO 论文证明了下界 $J(\pi') \ge L_\pi(\pi') - C\cdot \max_s D_{\text{KL}}(\pi\,\|\,\pi')$（$C$ 与 $\gamma$ 和最大优势有关）。只要每步把 KL 控制住，最大化代理目标就能**单调改进**。

## TRPO

$$\max_\theta\ \mathbb E\Big[\frac{\pi_\theta(a\mid s)}{\pi_{\text{old}}(a\mid s)}\hat A\Big] \quad \text{s.t.}\quad \mathbb E_s\big[D_{\text{KL}}(\pi_{\text{old}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s))\big] \le \delta$$

求解思路：目标做一阶展开 $g^\top\Delta\theta$，KL 做二阶展开 $\tfrac12\Delta\theta^\top F\Delta\theta$（$F$ 是 Fisher 信息矩阵），解得**自然梯度**方向

$$\Delta\theta = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1} g$$

$F^{-1}g$ 用共轭梯度 + Fisher-向量积求，再做回溯线搜索保证 KL 约束和目标确实改进。

> [!NOTE]
> 自然梯度的直觉：普通梯度在**参数空间**走固定步长，但参数小变化可能造成策略分布大变化（反之亦然）。自然梯度在**分布空间**（用 KL 度量）走固定步长，对参数化方式不敏感。代价是二阶计算，大模型上不可行，这正是 PPO 出现的原因。

## PPO-clip

PPO（Schulman et al., 2017）用一阶优化 + 比值截断近似 trust region。记 $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}$：

$$L^{\text{CLIP}}(\theta) = \mathbb E_t\Big[\min\big(r_t\hat A_t,\ \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big]$$

取 min 构造的是**悲观下界**：只在“对我们有利”的方向截断收益，不截断损失。

### 逐情况分析

| 情况 | $r_t$ 区间 | min 取到 | 梯度 | 含义 |
|---|---|---|---|---|
| $\hat A>0$ | $r \le 1+\epsilon$ | $r\hat A$ | 有 | 好动作，继续提高概率 |
| $\hat A>0$ | $r > 1+\epsilon$ | $(1+\epsilon)\hat A$ | **0** | 已经提得够多，停 |
| $\hat A<0$ | $r \ge 1-\epsilon$ | $r\hat A$ | 有 | 坏动作，继续压低 |
| $\hat A<0$ | $r < 1-\epsilon$ | $(1-\epsilon)\hat A$ | **0** | 已经压得够低，停 |

注意两个“没被截断”的角落：$\hat A>0$ 但 $r<1-\epsilon$（好动作概率反而降了），$\hat A<0$ 但 $r>1+\epsilon$（坏动作概率反而升了）。min 会选未截断项，梯度照常把它们拉回来。

框架里写的是 loss（目标取负），所以 min 变成 max：

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

> [!Q]
> **为什么 verl 的 PPO loss 是 `torch.maximum`，不是 min？**
>
> 论文最大化 $\min(r\hat A, \operatorname{clip}(r)\hat A)$，loss 是它的相反数：$-\min(x,y) = \max(-x,-y)$。所以“目标取 min”和“loss 取 max”是同一件事。面试时把这个等式写出来，再按 A 的正负讲上面的表就够了。

### dual-clip

$\hat A<0$ 且 $r$ 很大时，$-r\hat A$ 没有上界，一个异常 token 就能主导梯度。dual-clip（Ye et al., 2020, arXiv 1912.09729）对负优势再加一层上限：

$$\text{loss} = \min\big(\max(-r\hat A,\ -\operatorname{clip}(r)\hat A),\ -c\hat A\big),\quad \hat A<0$$

即 loss 最多是 $c\,\lvert\hat A\rvert$。verl 中 `actor.clip_ratio_c` 默认 3.0，且断言必须大于 1；对应监控指标 `actor/pg_clipfrac_lower`。

### clip-higher（非对称 clip）

对称 $\epsilon=0.2$ 时，一个概率 0.01 的 token 一次最多涨到 0.012，而概率 0.9 的 token 可以涨到 1.0 以上（被概率上限截住）。低概率的“探索性” token 很难被提上来，熵迅速下降。DAPO 把上界放宽：$\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$。

verl 配置：`actor.clip_ratio_low` / `actor.clip_ratio_high`，默认都是 0.2（与 `clip_ratio` 相同）。

> [!SRC]
> `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla`：先 `clamp(log_prob - old_log_prob, -20, 20)` 再 `exp` 得到 ratio；`pg_clipfrac` 统计被上/下界截断的比例；dual-clip 只在 `advantages < 0` 处生效（`torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)`）；最后交给 `agg_loss` 按 `loss_agg_mode` 聚合。

## 价值函数与 value clip

critic 回归 GAE 算出的 returns：$\mathcal L_V = \tfrac12(V_\phi(s_t) - \hat R_t)^2$。PPO 的常见实现还会对 value 做类似的 clip：

$$V^{\text{clip}} = V_{\text{old}} + \operatorname{clip}(V_\phi - V_{\text{old}}, -\epsilon_v, \epsilon_v),\qquad \mathcal L_V = \tfrac12\max\big((V_\phi-\hat R)^2,\ (V^{\text{clip}}-\hat R)^2\big)$$

verl 的 `compute_value_loss` 就是这个形式，`critic.cliprange_value` 默认 0.5。value clip 的必要性有争议（一些消融认为它帮助不大），但它限制了 critic 每步的变化量。

> [!WARN]
> value clip 用的是 $\max$：两个损失里取大的，同样是悲观的。不要和 policy clip 的方向搞混——policy 那边是“目标取 min / loss 取 max”，value 这边本来就是 loss，直接取 max。

## GAE

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\qquad \hat A_t = \sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}$$

反向递推 $\hat A_t = \delta_t + \gamma\lambda\hat A_{t+1}$，returns $=\hat A + V$ 作为 critic 目标，优势再做一次 masked whiten（零均值单位方差）给 actor 用。λ 越大越接近 MC（低偏差高方差）。LLM 场景常用 $\gamma=\lambda=1$。多轮 agent 中 observation token 被 mask，对递推“透明”。动手：exercises/model/gae。

## entropy bonus

$\mathcal L = \mathcal L^{\text{CLIP}} - c_H\, \mathcal H[\pi_\theta]$，鼓励探索。LLM-RL 中熵 bonus 容易引发熵爆炸（输出变乱码），verl 默认 `actor.entropy_coeff=0`，更常用 clip-higher 或 Clip-Cov / KL-Cov 控制熵（见“KL 与熵”一篇）。

## 一次 PPO 迭代的结构

<div class="flow"><span class="f-infra">rollout<small>train_batch_size 个 prompt × rollout.n</small></span><i>→</i><span>reward / old_log_prob / ref / values</span><i>→</i><span class="f-accent">compute_advantage</span><i>→</i><span class="f-rl">update_critic</span><i>→</i><span class="f-rl">update_actor<small>ppo_epochs × mini-batches</small></span></div>

- `data.train_batch_size`：每个 RL step 的 prompt 数；生成的样本数是它乘以 `rollout.n`。
- `actor.ppo_mini_batch_size`：每次**优化器 step** 用多少 prompt（全局数，verl 内部会再乘 `rollout.n`）。每个 RL step 的优化器步数 $= \text{ppo\_epochs}\times\frac{\text{train\_batch\_size}}{\text{ppo\_mini\_batch\_size}}$。
- `ppo_micro_batch_size_per_gpu`：梯度累积的切分单位，只影响显存，不应影响数学结果（前提是 loss 按全局 token 数归一化，见“实践中的坑”）。
- 只有第一个 mini-batch 的 ratio 恒为 1；之后的 mini-batch 都是轻度 off-policy，clip 开始起作用。若 `train_batch_size == ppo_mini_batch_size` 且 `ppo_epochs=1`，整个更新严格 on-policy，clip 永远不触发。
- **critic warmup**：`trainer.critic_warmup` 步之前只更新 critic、不更新 actor，避免用随机初始化的 value 头算出的优势去扰乱策略（V1 trainer 的 `_step_once` 里 `if critic_warmup <= global_steps: update_actor`）。

> [!Q]
> **pg_clipfrac 和 ppo_kl 分别说明什么？**
>
> `pg_clipfrac` 是被 clip 截断的 token 比例，持续偏高说明每步更新太猛（lr 大、epoch 多、mini-batch 多）或训推不一致；`ppo_kl` 是 $\pi_{\text{old}}$ 与 $\pi_\theta$ 的近似 KL（$-\overline{\log r}$）。两者都接近 0 说明根本没在 off-policy 区域训练，clip 形同虚设。

> [!Q]
> **PPO 相比 TRPO 丢了什么？**
>
> clip 不是严格的 KL 约束：它只让“越界 token 的梯度为零”，并不阻止参数被其他 token 的梯度带着走远，所以实际 KL 可能超出预期；多 epoch 时更明显。因此实践中常配合 early stopping（按 KL）、较小 lr、KL 正则。好处是只要一阶优化，能扩展到百亿参数。

## 速记

- 代理目标只在 trust region 内可信；TRPO 用 KL 约束 + 自然梯度，PPO 用 clip。
- 目标取 min = loss 取 max；A>0 截上界、A<0 截下界。
- dual-clip 给负优势加上限 $c$（verl 默认 3.0）；clip-higher 放宽上界（DAPO 0.28）。
- value clip：$\tfrac12\max$（verl `cliprange_value=0.5`）。
- 结构：rollout → 打分 → 优势 → critic 更新 → actor 多 mini-batch 更新。
- 动手：exercises/model/ppo_loss、exercises/model/gae。
