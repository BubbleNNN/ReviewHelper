---
title: 策略梯度与 REINFORCE
summary: 用 log-derivative trick 推出策略梯度定理，讲清 baseline、优势函数、重要性采样和方差从哪里来。
tags: RL, 策略梯度, REINFORCE
order: 20
---

## 目标与 log-derivative trick

参数化策略 $\pi_\theta$，目标是最大化期望回报：

$$J(\theta) = \mathbb E_{\tau \sim p_\theta}[R(\tau)], \qquad p_\theta(\tau) = \rho_0(s_0)\prod_t \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$$

难点在于期望的分布本身依赖 $\theta$。用恒等式 $\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$（log-derivative trick，也叫 score function / likelihood ratio）：

$$\nabla_\theta J = \sum_\tau R(\tau)\nabla_\theta p_\theta(\tau) = \mathbb E_{\tau\sim p_\theta}\big[R(\tau)\, \nabla_\theta \log p_\theta(\tau)\big]$$

轨迹概率取 log 后，环境转移 $P$ 和 $\rho_0$ 与 $\theta$ 无关，梯度为零，只剩策略项：

$$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

所以**不需要环境模型**，也不需要奖励可导。这是策略梯度能直接用在“答案对不对”这种离散奖励上的根本原因。

## 因果性与 reward-to-go

$t$ 时刻的动作影响不了 $t$ 之前的奖励，那部分在期望下与 $\nabla\log\pi(a_t\mid s_t)$ 的乘积为零，可以去掉：

$$\nabla_\theta J = \mathbb E\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\Big], \qquad G_t = \sum_{k\ge t}\gamma^{k-t} r_k$$

再把 $G_t$ 换成它的条件期望 $Q^\pi(s_t,a_t)$，就是**策略梯度定理**（Sutton et al., 2000）：

$$\nabla_\theta J = \mathbb E_{s\sim d^\pi,\, a\sim\pi_\theta}\big[Q^\pi(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)\big]$$

其中 $d^\pi$ 是策略诱导的（折扣）状态访问分布。

## baseline 为什么不改变期望

对任意只依赖状态、不依赖动作的函数 $b(s)$：

$$\mathbb E_{a\sim\pi}\big[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\big] = b(s)\sum_a \nabla_\theta \pi_\theta(a\mid s) = b(s)\,\nabla_\theta \sum_a \pi_\theta(a\mid s) = b(s)\nabla_\theta 1 = 0$$

所以把 $Q$ 换成 $Q - b$ 期望不变、方差可以大幅下降。取 $b = V^\pi(s)$ 就得到**优势形式**：

$$\nabla_\theta J = \mathbb E\big[A^\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\big]$$

> [!WARN]
> baseline 只能依赖状态（或与当前动作独立的量），不能依赖当前样本自己的动作或奖励。RLOO 用“组内**其他**样本”的平均奖励当 baseline，所以无偏；GRPO 用包含自身的组均值，严格说有一个 $\frac{1}{G}$ 量级的缩放偏差，再除以组内 std 又引入了与样本相关的缩放，这些都是 Dr.GRPO 讨论的问题。

方差最小的 baseline 并不是 $V(s)$，而是按 $\lVert\nabla\log\pi\rVert^2$ 加权的回报均值；但 $V(s)$ 简单、效果接近，是默认选择。

## REINFORCE

最朴素的蒙特卡洛策略梯度（Williams, 1992）：

```python
for iteration in range(N):
    trajs = sample(pi_theta)                     # on-policy 采样
    for (s_t, a_t, G_t) in trajs:
        grad += (G_t - b(s_t)) * grad_log_pi(a_t | s_t)
    theta += lr * grad / len(trajs)
```

LLM 上的序列级写法：对 prompt $x$ 采样 response $y$，

$$\nabla_\theta J \approx \frac1N\sum_{i}\big(R(x,y^{(i)}) - b(x)\big)\sum_{t}\nabla_\theta\log\pi_\theta(y^{(i)}_t \mid x, y^{(i)}_{\lt t})$$

GRPO、RLOO、ReMax、REINFORCE++ 本质上都是“REINFORCE + 不同的 baseline / 归一化 + PPO 式 clip”。

### 用 loss 实现梯度

框架里不会手写梯度，而是构造一个**代理损失**（surrogate loss），让它的梯度恰好等于策略梯度：

$$\mathcal L(\theta) = -\frac{1}{|\mathcal T|}\sum_t \hat A_t \log\pi_\theta(a_t\mid s_t)$$

其中 $\hat A_t$ 被当作常数（detach）。PPO 的版本把 $\log\pi_\theta$ 换成比值 $r_t(\theta) = \pi_\theta/\pi_{\text{old}}$：在 $\theta = \theta_{\text{old}}$ 处

$$\nabla_\theta r_t = \frac{\pi_\theta}{\pi_{\text{old}}}\nabla_\theta\log\pi_\theta = \nabla_\theta\log\pi_\theta$$

两者梯度相同；参数离开 $\theta_{\text{old}}$ 后，比值形式就是重要性采样修正后的目标。

> [!Q]
> **为什么损失里用 $\log\pi$ 而不是 $\pi$？**
>
> 这是 log-derivative trick 的直接结果：期望 $\mathbb E_\pi[f]$ 的梯度是 $\mathbb E_\pi[f\nabla\log\pi]$。数值上也更好：序列概率是一串小数相乘，取 log 变成求和，避免下溢；实现上用 `log_softmax` 再 `gather` 拿到每个 token 的 logprob（动手：exercises/model/logprob_entropy）。

## 重要性采样与 off-policy 修正

若数据来自行为策略 $\mu$，目标策略是 $\pi$：

$$\mathbb E_{a\sim\pi}[f(a)] = \mathbb E_{a\sim\mu}\Big[\frac{\pi(a)}{\mu(a)} f(a)\Big]$$

对整条轨迹，比值是逐步连乘 $\prod_t \frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}$，长度一长方差就爆炸。PPO/TRPO 的做法是只用**逐 token 比值**并忽略状态分布的变化：

$$L_{\pi_{\text{old}}}(\pi) = \mathbb E_{s\sim d^{\pi_{\text{old}}},\, a\sim\pi_{\text{old}}}\Big[\frac{\pi(a\mid s)}{\pi_{\text{old}}(a\mid s)} A^{\pi_{\text{old}}}(s,a)\Big]$$

这是真实目标的一阶近似，只有 $\pi$ 离 $\pi_{\text{old}}$ 足够近时才可信，所以需要 trust region（TRPO 的 KL 约束）或 clip（PPO）。

> [!KEY]
> 重要性采样能修正“动作分布”的偏移，但逐 token 比值不修正“状态分布”的偏移。这就是 PPO 必须把策略限制在旧策略附近的原因，也是异步 RL、训推不一致时需要额外截断（TIS）或拒绝（IcePop / rollout_rs）的原因。

## 方差从哪里来

1. **回报本身**：长 horizon、随机奖励、稀疏奖励（只有最后一步非零，前面所有 token 共享同一个信号）。
2. **采样**：每个 prompt 只采少量样本时，$R - b$ 的估计噪声大。GRPO 一个 prompt 采 $n$ 个，就是用样本数换方差。
3. **score function**：$\nabla\log\pi$ 对低概率动作的范数大。
4. **重要性比值**：偏离越远比值越大，方差越大。

常用降方差手段：baseline / critic、优势标准化（whiten）、更大的 batch、组采样、KL 正则、比值 clip。

> [!Q]
> **策略梯度为什么必须是 on-policy 的？**
>
> 推导里的期望是对 $p_\theta$ 取的，样本必须来自当前策略才是无偏估计。用旧数据就得乘重要性比值，比值的方差随偏离程度增长。PPO 允许小幅 off-policy（多 epoch、多 mini-batch），代价是用 clip 截断大比值带来的偏差。

> [!Q]
> **reward 全为正时不减 baseline 会怎样？**
>
> 期望梯度仍然正确，但每个被采样的动作概率都会被往上推，只是推的力度不同；靠“没被采样的动作被挤下去”间接区分好坏，方差很大、学得很慢。减 baseline 后差于平均的动作会被直接压低。

## 速记

- $\nabla J = \mathbb E[\hat A\, \nabla\log\pi]$，环境模型和奖励的可导性都不需要。
- baseline 与动作无关 ⇒ 期望不变；$V(s)$ 是常用 baseline，$A = Q - V$。
- 代理损失 $-\hat A\log\pi$ 或 $-\hat A\, r(\theta)$，二者在 $\theta_{\text{old}}$ 处梯度相同。
- 重要性采样修正动作分布、不修正状态分布 ⇒ 需要 trust region。
- 动手：exercises/model/logprob_entropy、exercises/model/ppo_loss。
