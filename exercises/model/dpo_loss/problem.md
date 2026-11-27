---
title: DPO 损失（含 label smoothing）
difficulty: 易
tags: RLHF, DPO, 偏好学习
order: 32
source: Rafailov et al. 2023 DPO；TRL DPOTrainer 的 sigmoid loss
---

输入是一个 batch 的偏好对，每个量都是 `(B,)` 的 `torch.Tensor`（测试用 float64），表示**序列级** log 概率（已经对 response token 求和）：

```python
dpo_loss(policy_chosen_logps, policy_rejected_logps,
         ref_chosen_logps, ref_rejected_logps,
         beta=0.1, label_smoothing=0.0) -> (loss, chosen_rewards, rejected_rewards, reward_accuracy)
```

$$h = \beta\Big[(\log\pi_\theta(y_w)-\log\pi_{ref}(y_w)) - (\log\pi_\theta(y_l)-\log\pi_{ref}(y_l))\Big]$$

$$\mathcal L = \text{mean}\Big[-(1-\varepsilon)\log\sigma(h) - \varepsilon\log\sigma(-h)\Big]$$

- `chosen_rewards = β (log π_θ(y_w) − log π_ref(y_w))`，`rejected_rewards` 同理，形状 `(B,)` 的 tensor，只用于日志，要 `.detach()`。
- `reward_accuracy`：`chosen_rewards > rejected_rewards` 的比例（python float）。
- `loss` 是 0-d tensor，之后要 `loss.backward()`。梯度只流向两个 `policy_*` 输入；参考模型是冻结的，`ref_*` 视为常数（`.detach()`）。
- $h$ 可能非常大或非常小，$\log\sigma$ 要数值稳定。
- $\varepsilon$ 就是 `label_smoothing`（cDPO：假设标注有 $\varepsilon$ 的概率是反的）。

<!-- solution -->

### 讲解

**推导主线**：KL 约束下的 RL 目标

$$\max_\pi \mathbb E[r(x,y)] - \beta\,\text{KL}(\pi\|\pi_{ref})$$

的最优解是 $\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)e^{r(x,y)/\beta}$。反解出 $r = \beta\log\frac{\pi^*}{\pi_{ref}} + \beta\log Z(x)$，代入 Bradley-Terry 偏好模型 $P(y_w\succ y_l)=\sigma(r_w-r_l)$，同一个 prompt 的 $\beta\log Z(x)$ 正好抵消，得到只依赖策略和参考模型的损失。这就是“策略本身是一个隐式奖励模型”。

**数值稳定**：$\log\sigma(h) = -\text{softplus}(-h) = -\log(1+e^{-h})$，用 `F.logsigmoid(h)`（或 `-F.softplus(-h)`）计算，$h=\pm 10^4$ 都不会溢出。`torch.log(torch.sigmoid(h))` 在 $h=-10^4$ 时 sigmoid 下溢成 0，loss 变成 inf。

**梯度直觉**：

$$\nabla\mathcal L = -\beta\,\sigma(-h)\big[\nabla\log\pi(y_w) - \nabla\log\pi(y_l)\big]$$

权重 $\sigma(-h)$ 在模型“排错了”时大、“已经排对”时小，相当于自带困难样本挖掘。batch 取平均后，对每个样本 chosen 侧的梯度就是 $-\beta\,\sigma(-h_i)/B$。加 label smoothing 后 $\partial\ell/\partial h = -(1-\varepsilon)\sigma(-h)+\varepsilon\,\sigma(h)$，在 $\sigma(h)=1-\varepsilon$ 处为 0，也就是 margin 不再被无限推大。

**detach 什么**：参考模型的 logp 在 TRL 里是 `no_grad` 下算出来的，手写时显式 detach 更保险；`chosen_rewards` / `rejected_rewards` 只用于日志，TRL 里也是 `.detach()` 过的，否则日志张量会一直挂着计算图。

**β 的作用**：β 越大，越强地约束策略留在参考模型附近（隐式奖励的尺度越大，sigmoid 更快饱和）。常用 0.1 左右。

> [!Q]
> DPO 训练时 chosen 的 logprob 也在下降，正常吗？
> 很常见。DPO 只优化两者的**差**，模型可以通过让 rejected 降得更快来降低损失，导致 chosen 和 rejected 的似然一起下降（likelihood displacement）。缓解办法：在 chosen 上加 SFT（NLL）损失（如 RPO）、调大 β、或者用 IPO 等对 margin 有上界的变体。

> [!Q]
> DPO 和 PPO-RLHF 的根本区别？
> DPO 是离线的：只在固定的偏好数据上学，不采样新回答，不需要奖励模型和 critic，简单稳定；PPO 是在线的：用当前策略采样，由奖励模型打分，能探索到数据集之外的回答，但工程复杂得多。在线 / 迭代式 DPO 是两者的折中。

```python
# TRL DPOTrainer 里 sigmoid loss 的写法
logits = beta * ((pc - rc) - (pr - rr))
loss = -(1 - ls) * F.logsigmoid(logits) - ls * F.logsigmoid(-logits)
```
