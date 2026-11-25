---
title: MDP 与价值函数
summary: 从 MDP、回报、V/Q/A、Bellman 方程讲到 MC 与 TD，最后把 LLM 生成写成一个 token 级 MDP。
tags: RL, 基础, Bellman
order: 10
---

## 马尔可夫决策过程

一个 MDP 由五元组 $(\mathcal S, \mathcal A, P, R, \gamma)$ 加初始分布 $\rho_0$ 描述：

- $\mathcal S$ 状态空间，$\mathcal A$ 动作空间；
- $P(s' \mid s, a)$ 转移概率，**马尔可夫性**指下一状态只依赖当前状态和动作；
- $R(s,a)$ 或 $r_t$ 即时奖励；
- $\gamma \in [0,1]$ 折扣因子。

策略 $\pi(a \mid s)$ 给出动作分布。与环境交互产生轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$，从 $t$ 时刻起的**回报**（return）是

$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

RL 的目标是最大化期望回报 $J(\pi) = \mathbb E_{\tau \sim \pi}[G_0]$。

## 三个价值函数

| 名称 | 定义 | 含义 |
|---|---|---|
| 状态价值 $V^\pi(s)$ | $\mathbb E_\pi[G_t \mid s_t = s]$ | 从 s 出发按 π 走，平均能拿多少 |
| 动作价值 $Q^\pi(s,a)$ | $\mathbb E_\pi[G_t \mid s_t=s, a_t=a]$ | 先做 a 再按 π 走 |
| 优势 $A^\pi(s,a)$ | $Q^\pi(s,a) - V^\pi(s)$ | 动作 a 比“平均水平”好多少 |

两条恒等式要记住：$V^\pi(s) = \mathbb E_{a\sim\pi}[Q^\pi(s,a)]$，因此 $\mathbb E_{a\sim\pi}[A^\pi(s,a)] = 0$。优势的期望为零，这正是策略梯度里“减 baseline 不改变期望、只降方差”的来源。

### Bellman 期望方程

把回报拆成“一步奖励 + 折扣后的剩余回报”：

$$V^\pi(s) = \mathbb E_{a\sim\pi,\, s'\sim P}\big[r(s,a) + \gamma V^\pi(s')\big]$$

$$Q^\pi(s,a) = r(s,a) + \gamma\, \mathbb E_{s'\sim P,\, a'\sim\pi}\big[Q^\pi(s',a')\big]$$

### Bellman 最优方程

最优价值 $V^\star(s) = \max_\pi V^\pi(s)$ 满足

$$Q^\star(s,a) = r(s,a) + \gamma\, \mathbb E_{s'}\big[\max_{a'} Q^\star(s',a')\big]$$

Bellman 算子在 $\gamma<1$ 时是 $\gamma$-压缩映射，所以价值迭代收敛到唯一不动点。Q-learning、DQN 就是在采样意义下做这个不动点迭代。

> [!NOTE]
> 方法谱系：**value-based**（学 $Q^\star$，再 $\arg\max$，如 DQN）、**policy-based**（直接对 $\pi_\theta$ 做梯度上升，如 REINFORCE）、**actor-critic**（策略 + 价值函数做 baseline，如 PPO）。LLM 的动作空间是整个词表（十万量级），且需要采样多样性，所以主流是 policy-based / actor-critic。

## MC 与 TD

估计 $V^\pi$ 有两条路：

<div class="cols">
<div><h4>Monte Carlo</h4><p>用完整回报 \(G_t\) 做目标：\(V(s_t) \leftarrow V(s_t) + \alpha (G_t - V(s_t))\)。</p><p>无偏；方差大（累积了整条轨迹的随机性）；必须等 episode 结束。</p></div>
<div><h4>TD(0)</h4><p>用一步 bootstrap 做目标：\(r_t + \gamma V(s_{t+1})\)。TD 误差 \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\)。</p><p>方差小；有偏（依赖当前 V 的准确度）；可在线更新。</p></div>
</div>

中间态是 n-step 回报和 TD(λ)：把 1 步、2 步、…、∞ 步的目标按 $(1-\lambda)\lambda^{n-1}$ 加权平均。把这个思想用在优势估计上就是 **GAE**：

$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k \ge 0} (\gamma\lambda)^k \delta_{t+k}$$

λ=0 退化为 TD 误差，λ=1 退化为 MC 回报减 $V$。动手：exercises/model/gae。

## on-policy 与 off-policy

- **行为策略**（behavior policy）：实际产生数据的策略；**目标策略**（target policy）：正在被优化的策略。
- 两者相同叫 on-policy（REINFORCE、A2C、严格意义上的 PPO 第一个 mini-batch），不同叫 off-policy（Q-learning、DQN 的 replay buffer、异步 RL）。
- off-policy 的数据要么用不依赖行为策略的目标（Q-learning 的 max 目标），要么用**重要性采样**比值 $\pi_\theta / \pi_{\text{behavior}}$ 纠偏。

> [!Q]
> **PPO 是 on-policy 还是 off-policy？为什么它还需要重要性比值？**
>
> PPO 是 on-policy 算法，但一批 rollout 会被切成多个 mini-batch、跑多个 epoch。第二个 mini-batch 开始，参数已经变了，数据相对当前策略就“轻度 off-policy”。比值 $r_t = \pi_\theta / \pi_{\text{old}}$ 加 clip 正是为了在这种小偏移下安全复用数据。异步 RL（verl 的 one-step-off、fully async）会让偏移更大，需要额外的 rollout correction。

## 偏差-方差权衡

RL 里几乎每个设计都是在偏差和方差之间挪动：

| 手段 | 方差 | 偏差 |
|---|---|---|
| 用 MC 回报 | 大 | 无 |
| 用 TD / critic bootstrap | 小 | 取决于 critic 质量 |
| 减 baseline | 降低 | 不引入（baseline 与动作无关时） |
| GAE 的 λ 从 1 调到 0 | 逐渐降低 | 逐渐增大 |
| 折扣 γ<1 | 降低 | 相对“无折扣目标”有偏 |
| 组内均值做 baseline（GRPO / RLOO） | 降低 | RLOO 无偏；GRPO 的 std 归一化有偏 |

## 把 LLM 生成写成 MDP

给定 prompt $x$，策略 $\pi_\theta$ 逐 token 生成 response $y = (y_1, \dots, y_T)$：

<div class="flow"><span>状态 \(s_t\)<small>prompt + 已生成前缀 \((x, y_{&lt;t})\)</small></span><i>→</i><span class="f-rl">动作 \(a_t\)<small>下一个 token \(y_t \in \mathcal V\)</small></span><i>→</i><span>转移<small>确定性拼接 \(s_{t+1} = s_t \oplus y_t\)</small></span><i>→</i><span class="f-accent">奖励<small>通常只在 EOS 处给 \(R(x,y)\)</small></span></div>

几个特点决定了 LLM-RL 的算法选择：

1. **转移是确定性的**。$s_{t+1}$ 完全由 $s_t, a_t$ 决定，于是 $Q(s_t,a_t) = r_t + \gamma V(s_{t+1})$，不需要对环境求期望。
2. **奖励稀疏**。outcome reward（答案对不对、RM 打分）只出现在最后一个 token；中间 token 的奖励只有可选的 KL 惩罚 $-\beta \log\frac{\pi}{\pi_{\text{ref}}}$。
3. **动作空间巨大**（词表 $10^5$ 量级），无法做 $\max_a Q$，所以用策略梯度。
4. **episode 很短但不定长**。以 EOS 或 max_response_length 截断；截断不是真正的终止，严格来说应该 bootstrap，实践里常直接当终止或做 overlong 惩罚。
5. **两种粒度**。token 级 MDP（PPO + critic，每个 token 一个优势）或把整个 response 当一个动作的 **contextual bandit**（GRPO、RLOO、ReMax 给整条序列同一个优势）。
6. **多轮 agent**。工具返回的 observation token 不是策略生成的，在 loss 和 GAE 里都要 mask 掉（verl 的 response_mask 对它们为 0）。

> [!Q]
> **为什么 LLM RL 里 γ 常取 1？**
>
> response 长度有限（几百到几万 token），奖励只在末尾，γ<1 会让早期 token 的信号被 $\gamma^{T}$ 指数衰减，等于惩罚长推理。verl 默认 `algorithm.gamma=1.0, lam=1.0`。有限长度保证 γ=1 时回报也有界。

> [!Q]
> **转移确定性对优势估计有什么影响？**
>
> $A(s_t, a_t) = r_t + V(s_{t+1}) - V(s_t)$ 就是 TD 误差本身，不再有“环境随机性”这一项方差来源；方差主要来自策略采样（后续 token 的随机性）和 critic 误差。也因此 critic 学得不准时，λ 取 1（纯 MC）往往更稳。

> [!SRC]
> verl 的 token 级视角：`token_level_rewards` 形状是 `(bs, response_length)`，outcome 分数只写在最后一个有效 token 上；`response_mask` 标出策略生成的 token。见 `verl/trainer/ppo/core_algos.py` 中 `compute_gae_advantage_return` 与 `compute_grpo_outcome_advantage` 的入参。

## 速记

- $A = Q - V$，$\mathbb E_\pi[A]=0$。
- Bellman 期望方程用于策略评估，最优方程用于控制。
- MC 无偏高方差，TD 有偏低方差，GAE 用 λ 在两者间插值。
- LLM 生成 = 确定性转移 + 稀疏终点奖励 + 巨大动作空间的 MDP，或者整条序列一个动作的 bandit。
