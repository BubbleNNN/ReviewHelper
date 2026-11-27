---
title: GAE 广义优势估计（带 mask）
difficulty: 中
tags: RL, PPO, verl
order: 20
source: verl/trainer/ppo/core_algos.py · compute_gae_advantage_return
---

实现 PPO 里的 GAE。输入都是 `(B, T)` 的 `torch.Tensor`，`T` 是 response 长度：

- `rewards[b, t]`：token 级奖励。outcome reward 通常只在最后一个有效 token 上非零。
- `values[b, t]`：critic 对每个位置的估值 $V(s_t)$。
- `mask[b, t]`：bool。True 表示这个 token 由策略生成、参与训练；False 表示 padding（EOS 之后）或多轮对话里的工具返回（observation）token。

公式：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$$

最后一个有效 token 之后 $V=0$、$\hat A=0$。返回 `(advantages, returns)`，其中 `returns = advantages + values`。

**mask 规则（和 verl 一致）**：从后往前扫，遇到 `mask=0` 的位置时，`nextvalue` 和 `lastgaelam` 都**原样跳过**，不被这个位置更新。也就是说 observation token 对优势计算是透明的，前一个有效 token 直接接到后一个有效 token 上。

```python
compute_gae(rewards, values, mask, gamma=1.0, lam=1.0) -> (adv, ret)
```

不需要做 whiten（标准化），测试只检查 `mask=True` 的位置。

两个返回值都**不能带梯度**：`values` 来自 critic，可能 `requires_grad=True`；`returns` 之后要当 value loss 的回归目标。用 `torch.no_grad()` 或 `.detach()`。输出 dtype 与输入一致（测试用 float64）。

<!-- solution -->

### 讲解

- **λ 的两个极端**：λ=0 时 $\hat A_t=\delta_t$，是单步 TD，方差小、偏差大（依赖 critic 准不准）；λ=1 时 $\hat A_t=\sum_k \gamma^k r_{t+k} - V(s_t)$，就是蒙特卡洛回报减 baseline，无偏但方差大。
- **LLM 场景的默认值**：verl 里 `algorithm.gamma=1.0, lam=1.0`。因为奖励稀疏（只在最后一个 token），response 又不长，所以不打折。
- **mask 的意义**：多轮 agent RL 里，工具返回的 token 不是策略生成的，不应该有优势，也不该打断价值的传递。verl 用 `nextvalues = V_t*m_t + (1-m_t)*nextvalues` 这种写法让它们“透明”。
- **之后的 whiten**：verl 在 GAE 之后会调用 `masked_whiten` 对优势做零均值、单位方差（只在有效 token 上统计），这样学习率对奖励尺度不敏感。
- **为什么不带梯度**：verl 的 `compute_gae_advantage_return` 整个包在 `torch.no_grad()` 里。如果 returns 带着 values 的梯度，value loss $(V-R)^2$ 会同时把目标往预测拉，等于没有目标。
- **bool mask**：`1 - mask` 对 bool tensor 会报错，先 `mask.to(rewards.dtype)`。
- **复杂度**：一次反向扫描，$O(BT)$。时间维有依赖，没法直接向量化，但 batch 维可以向量化。
