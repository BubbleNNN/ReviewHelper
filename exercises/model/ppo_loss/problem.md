---
title: PPO 策略损失（clip-higher、dual-clip、loss_agg_mode）
difficulty: 中
tags: RL, PPO, verl
order: 26
source: verl/trainer/ppo/core_algos.py · compute_policy_loss_vanilla + agg_loss
---

按 verl 的 `compute_policy_loss_vanilla` 和 `agg_loss`（单卡，`dp_size=1`）实现 PPO 的策略损失。所有输入形状都是 `(B, T)`。

```python
agg_loss(loss_mat, loss_mask, loss_agg_mode, loss_scale_factor=None) -> float
ppo_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0,
                loss_agg_mode="token-mean") -> (loss, metrics)
```

**逐 token 损失**（$r=\exp(\log\pi-\log\pi_{old})$，指数之前先把差值 clamp 到 $[-20, 20]$）：

$$L_1=-A r,\quad L_2=-A\,\text{clip}(r, 1-\epsilon_{low}, 1+\epsilon_{high}),\quad L^{clip}=\max(L_1,L_2)$$

$$L = \begin{cases} \min(L^{clip},\ -A\cdot c) & A<0 \quad\text{(dual-clip)}\\ L^{clip} & A\ge 0\end{cases}$$

**`metrics`**（都是对 `response_mask` 的 masked mean，返回 Python float）：

- `pg_clipfrac`：$L_2 > L_1$ 的比例
- `ppo_kl`：$-(\log\pi-\log\pi_{old})$（clamp 之后的值）的均值
- `pg_clipfrac_lower`：$A<0$ 且 $L^{clip} > -A c$ 的比例

**四种 `loss_agg_mode`**（被 mask 掉的位置不参与；一整行都被 mask 的序列不计入序列数）：

| 模式 | 计算 |
|---|---|
| `token-mean` | 所有有效 token 的损失和 / 有效 token 数 |
| `seq-mean-token-sum` | 每条序列内求和，再对有效序列取平均 |
| `seq-mean-token-mean` | 每条序列内取平均（分母加 `1e-8`），再对有效序列取平均 |
| `seq-mean-token-sum-norm` | `seq-mean-token-sum` 的结果再除以 `loss_scale_factor`（默认 `T`，即 `loss_mask.shape[-1]`） |

其他模式名抛 `ValueError`。

<!-- solution -->

### 讲解

**clip 的直觉**（$\epsilon=0.2$）：

| 情况 | 结果 |
|---|---|
| $A>0$，$r>1.2$ | 取 $L_2$，梯度为 0：好动作的概率已经涨够了，不再推 |
| $A>0$，$r<1$ | 取 $L_1$，正常梯度：概率下降的好动作要拉回来 |
| $A<0$，$r<0.8$ | 取 $L_2$，梯度为 0：坏动作的概率已经降够了 |
| $A<0$，$r>1$ | 取 $L_1$，正常梯度，而且 $r$ 越大损失越大 |

**dual-clip**：最后一种情况下 $r$ 可能非常大（比如 off-policy 样本），$-A r$ 无上界，一个 token 就能主导整个 batch 的梯度。dual-clip 把它截在 $-A c$（$c=3$），超过后梯度为 0。

**clip-higher**（DAPO）：把 $\epsilon_{high}$ 调大（如 0.28），给低概率的好 token 更多上涨空间，缓解 entropy collapse。verl 里对应 `clip_ratio_low` / `clip_ratio_high`。

**loss_agg_mode 的偏差**

- `token-mean`：长回答的每个 token 和短回答的每个 token 权重相同，总体上长回答贡献更多梯度。verl 默认，DAPO 推荐。
- `seq-mean-token-mean`：原始 GRPO 的写法，每条序列权重相同，但长序列里每个 token 被稀释。答错时倾向于“写长一点来稀释惩罚”。
- `seq-mean-token-sum-norm`：Dr.GRPO 的写法，用一个常数代替序列长度做归一化，去掉长度偏差。

**多卡时**：verl 里 `token-mean` 用的是**全局** token 数（`batch_num_tokens`）而不是本卡 token 数，再乘 `dp_size` 抵消 DDP/FSDP 的梯度平均，这样损失和并行方式无关。

> [!Q]
> 为什么 clipfrac 是一个重要的监控指标？
> 它反映新旧策略偏离的程度。clipfrac 持续很高（比如 >0.2）说明每次更新步子太大或者数据太 off-policy，大量 token 没有梯度，学习效率下降，可以调小学习率或减少 PPO epoch；接近 0 则说明 clip 没起作用，可能可以更激进一点。

> [!Q]
> 为什么 ratio 要在 log 空间做差再 exp，并且 clamp 到 ±20？
> logprob 本来就是 log 空间的量，相减再 exp 比两个概率相除稳定；clamp 防止异常样本让 $e^{x}$ 溢出成 inf，进而产生 nan 梯度。
