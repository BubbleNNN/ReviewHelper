---
title: RL 基础
summary: MDP、价值函数、策略梯度、TRPO/PPO、GAE 的高频面试题。
tags: RL
order: 10
---

## 什么是 MDP？马尔可夫性指什么？

五元组 $(\mathcal S, \mathcal A, P, R, \gamma)$ 加初始分布。马尔可夫性：下一状态与奖励只依赖当前状态和动作，$P(s_{t+1}\mid s_t,a_t,s_{t-1},\dots) = P(s_{t+1}\mid s_t,a_t)$。

LLM 生成里把“prompt + 已生成前缀”当状态，就天然满足马尔可夫性。

## V、Q、A 的定义和关系？为什么优势的期望是 0？

- $V^\pi(s) = \mathbb E_\pi[G_t\mid s_t=s]$
- $Q^\pi(s,a) = \mathbb E_\pi[G_t\mid s_t=s,a_t=a]$
- $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$

因为 $V^\pi(s) = \mathbb E_{a\sim\pi}[Q^\pi(s,a)]$，所以 $\mathbb E_{a\sim\pi}[A^\pi(s,a)] = 0$。优势衡量“这个动作比平均好多少”。

## 写出 Bellman 期望方程和最优方程

期望方程（策略评估）：

$$V^\pi(s) = \mathbb E_{a\sim\pi, s'}\big[r(s,a) + \gamma V^\pi(s')\big]$$

最优方程（控制）：

$$Q^\star(s,a) = r(s,a) + \gamma\,\mathbb E_{s'}\big[\max_{a'}Q^\star(s',a')\big]$$

$\gamma<1$ 时 Bellman 算子是压缩映射，迭代收敛到唯一不动点。

## MC 和 TD 有什么区别？偏差和方差分别如何？

- MC 用完整回报 $G_t$ 做目标：无偏、方差大、要等 episode 结束。
- TD(0) 用 $r_t + \gamma V(s_{t+1})$ 做目标（bootstrap）：方差小、有偏（依赖 V 的准确度）、可在线更新。
- n-step / TD(λ) / GAE 在两者之间插值。

## on-policy 和 off-policy 的区别？

看产生数据的行为策略和被优化的目标策略是否相同。on-policy：REINFORCE、A2C、PPO（近似）；off-policy：Q-learning、DQN、SAC。off-policy 可以复用旧数据、样本效率高，但需要重要性采样或与行为策略无关的目标（如 max 算子）来纠偏。

## Q-learning 为什么是 off-policy 的？

更新目标 $r + \gamma\max_{a'}Q(s',a')$ 对应的是贪心目标策略，与实际用 ε-greedy 等行为策略选了什么动作无关。所以任何策略产生的 $(s,a,r,s')$ 都能用，这也是 DQN 能用 experience replay 的前提。

## DQN 的两个关键技巧是什么？

1. **Experience replay**：打乱样本，打破时间相关性，提高数据复用。
2. **Target network**：TD 目标用一个延迟更新的网络计算，避免“追着自己跑”导致发散。

常见改进：Double DQN（用在线网络选动作、目标网络估值，缓解过估计）、Dueling、Prioritized replay。

## 推导策略梯度

$$\nabla J = \nabla\sum_\tau p_\theta(\tau)R(\tau) = \sum_\tau p_\theta(\tau)\nabla\log p_\theta(\tau)R(\tau)$$

$\log p_\theta(\tau)$ 中环境转移与 $\theta$ 无关，只剩 $\sum_t\nabla\log\pi_\theta(a_t\mid s_t)$。再利用因果性把 $R(\tau)$ 换成 reward-to-go $G_t$，得到 $\nabla J = \mathbb E[\sum_t G_t\nabla\log\pi_\theta(a_t\mid s_t)]$。

## 为什么减 baseline 不改变梯度期望？

$$\mathbb E_{a\sim\pi}[b(s)\nabla\log\pi(a\mid s)] = b(s)\sum_a\nabla\pi(a\mid s) = b(s)\nabla 1 = 0$$

前提是 $b$ 不依赖当前动作（也不依赖当前样本的奖励）。它不改变期望，但能大幅降低方差；$V(s)$ 是最常用的 baseline，于是得到优势形式。

## REINFORCE 有什么缺点？怎么改进？

方差大（MC 回报 + 稀疏奖励）、样本效率低（严格 on-policy，每批数据只用一次）、对奖励尺度敏感。改进：减 baseline / 用 critic（actor-critic）、优势标准化、GAE、多 epoch 复用数据并用 PPO-clip 保证稳定、同 prompt 多采样（GRPO / RLOO）。

## 重要性采样是什么？在 PPO 里怎么用？

$\mathbb E_{p}[f] = \mathbb E_q[\frac{p}{q}f]$，用 $q$ 的样本估计 $p$ 下的期望。PPO 用逐 token 比值 $r_t = \pi_\theta/\pi_{\text{old}}$ 复用旧策略的样本。问题：比值偏离 1 时方差爆炸；整条轨迹的连乘比值更不可用。所以 PPO 只用逐 token 比值并 clip，忽略状态分布的偏移。

## TRPO 的核心思想？

性能差引理给出代理目标 $\mathbb E[\frac{\pi'}{\pi}A^\pi]$，它只在新旧策略接近时可信。TRPO 最大化代理目标，同时约束平均 KL 不超过 $\delta$，从而保证近似单调改进。求解：目标一阶展开、KL 二阶展开（Fisher 矩阵），共轭梯度求自然梯度方向，再线搜索。

## 自然梯度是什么？为什么大模型不用？

在 KL 度量的分布空间里走固定步长：$\Delta\theta \propto F^{-1}g$，$F$ 是 Fisher 信息矩阵。它对参数化方式不敏感，更新更稳定。大模型参数量太大，即使用共轭梯度 + Fisher-向量积也很贵，所以 PPO 用一阶优化 + clip 近似 trust region。

## 写出 PPO-clip 目标，并按优势正负解释

$$L = \mathbb E\big[\min(r\hat A,\ \operatorname{clip}(r,1-\epsilon,1+\epsilon)\hat A)\big]$$

- $\hat A>0$：$r>1+\epsilon$ 后被截断、梯度为 0，防止好动作概率涨太多；
- $\hat A<0$：$r<1-\epsilon$ 后被截断，防止坏动作概率压太狠；
- 反方向偏离（好动作概率反而降了等）不截断，梯度照常把它拉回来。取 min 是悲观下界。

## 为什么 PPO 的 clip 在 loss 里写成 max？

论文最大化 $\min(r\hat A, \operatorname{clip}(r)\hat A)$，框架最小化 loss $= -\min(\cdot) = \max(-r\hat A, -\operatorname{clip}(r)\hat A)$。对 $\hat A<0$ 也一样：loss 取两者较大的那个，等价于目标取较小的那个，都是悲观估计。

## dual-clip 解决什么问题？

$\hat A<0$ 且 $r$ 很大时，loss $-r\hat A$ 无上界，个别 token 可以主导梯度。dual-clip 在负优势上再加上限：loss 不超过 $c\lvert\hat A\rvert$。verl 的 `clip_ratio_c` 默认 3.0，必须大于 1。

## GAE 的公式和 λ 的作用？

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\quad \hat A_t = \sum_k(\gamma\lambda)^k\delta_{t+k}$$

反向递推 $\hat A_t = \delta_t + \gamma\lambda\hat A_{t+1}$。λ=0 是单步 TD（低方差高偏差），λ=1 是 MC 回报减 V（无偏高方差）。critic 目标是 returns $=\hat A + V$。

## value clip 是什么？

$V^{\text{clip}} = V_{\text{old}} + \operatorname{clip}(V - V_{\text{old}}, \pm\epsilon_v)$，loss $= \tfrac12\max((V-R)^2, (V^{\text{clip}}-R)^2)$，限制 critic 每次更新的变化量。verl `critic.cliprange_value` 默认 0.5。有研究认为它作用有限，但它是 PPO 常见实现细节。

## critic 的作用是什么？LLM 里为什么常去掉？

critic 估计 $V(s)$ 作为 baseline 并提供 token 级优势（通过 GAE），降低方差、做信用分配。LLM 中 critic 与 actor 同规模，显存和计算翻倍；稀疏 outcome reward 下 token 级 value 难学准，还需要 warmup。GRPO / RLOO 用同 prompt 多样本的奖励均值替代它。

## 熵正则有什么作用？LLM-RL 里为什么默认关掉？

loss 减去 $c_H\mathcal H$，鼓励探索、防止过早收敛。LLM 词表巨大，熵 bonus 一旦偏大会让分布变平，输出乱码、长度暴涨（熵爆炸），且系数很敏感。verl 默认 `entropy_coeff=0`，更常用 clip-higher、Clip-Cov / KL-Cov 管理熵。

## 折扣因子 γ 有什么作用？LLM 里为什么常取 1？

γ 让无限 horizon 的回报有界，并降低远期回报带来的方差（代价是偏向短期）。LLM response 长度有限，奖励只在末尾；γ<1 会让长推理开头的 token 几乎收不到信号。verl 默认 `gamma=1.0, lam=1.0`。

## PPO 里 ratio 为什么用 exp(logp − old_logp) 算？

直接相除两个很小的概率会下溢、精度差；logprob 相减再 exp 数值稳定。verl 还会先把差值 clamp 到 $[-20, 20]$ 再 exp，防止溢出。$\text{ppo\_kl}$ 指标就是 $-(\log\pi_\theta - \log\pi_{\text{old}})$ 的 masked 均值。

## LLM RL 怎么做探索？

主要靠采样：温度 1.0 左右的随机采样、每个 prompt 采多条（rollout.n）。辅助手段：clip-higher 让低概率 token 能涨起来、控制熵塌缩、数据侧挑选“有时对有时错”的题、动态采样。经典 RL 的 ε-greedy 之类在词表级别没有意义。
