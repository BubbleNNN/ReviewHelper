---
title: KL 估计量 k1 / k2 / k3
difficulty: 中
tags: RL, KL, verl
order: 30
source: verl/trainer/ppo/core_algos.py · kl_penalty_forward；Schulman, Approximating KL Divergence (2020)
---

RL 里我们只有从当前策略 $\pi$ 采样得到的 token 和两边的 logprob，没法对整个 vocab 求精确 KL。于是用单样本估计量近似 $\text{KL}(\pi\,\|\,\pi_{ref}) = \mathbb E_{x\sim\pi}[\log\pi(x) - \log\pi_{ref}(x)]$。

输入都是 `torch.Tensor`（测试用 float64）：

```python
kl_penalty(logprob, ref_logprob, kind)   # 逐元素，形状、dtype 与输入相同，要能对 logprob 反传
exact_kl(p, q) -> 0-d tensor             # 两个离散分布 (V,) 的精确 KL(p||q)，约定 0·log0 = 0
```

`kind` 对齐 verl 的 `kl_penalty_forward`：

| kind | 公式（$\ell=\log\pi$，$\ell_{ref}=\log\pi_{ref}$） |
|---|---|
| `"kl"` / `"k1"` | $\ell-\ell_{ref}$ |
| `"abs"` | $\lvert\ell-\ell_{ref}\rvert$ |
| `"mse"` / `"k2"` | $\frac12(\ell-\ell_{ref})^2$ |
| `"low_var_kl"` / `"k3"` | 令 $u=\text{clip}(\ell_{ref}-\ell, -20, 20)$，结果为 $\text{clip}(e^{u}-u-1,\ -10,\ 10)$ |

其他 `kind` 抛 `ValueError`。clamp 用 `torch.clamp`，全程用 torch 运算，因为 k3 会直接当 loss 反传。`exact_kl` 请自己写，**不要直接调用** `F.kl_div` 或 `torch.distributions.kl_divergence`（测试会拿它们当参考答案）。

测试会从 $\pi$ 采样，用蒙特卡洛检查估计量的期望和方差，还会检查对 `logprob` 的梯度。

<!-- solution -->

### 讲解

记 $r = \pi_{ref}(x)/\pi(x)$，$x\sim\pi$。

- **k1** $=-\log r$：$\mathbb E[k_1]=\text{KL}$，**无偏**，但单个样本可以是负数，方差大。
- **k2** $=\frac12(\log r)^2$：总是非负、方差小，但**有偏**；当两个分布很接近时，它和 KL 只差三阶小量。
- **k3** $=(r-1)-\log r$：因为 $\mathbb E_\pi[r]=\sum_x \pi_{ref}(x)=1$，所以 $\mathbb E[k_3]=\mathbb E[-\log r]=\text{KL}$，**无偏**；又因为 $\log r \le r-1$，每个样本都 $\ge 0$。分布接近时方差也比 k1 小。

**梯度要另外考虑**：上面讲的都是“估计 KL 的值”。把它们当 loss 求导时，样本 $x$ 是固定的，autograd 只对 $\log\pi(x)$ 求导，不管“采样分布本身也依赖 $\theta$”这一项。逐元素导数是 $\partial k_1/\partial\ell=1$、$\partial k_2/\partial\ell=\ell-\ell_{ref}$、$\partial k_3/\partial\ell=1-r$，于是

- $\mathbb E_\pi[\nabla k_1]=\sum_x\pi\nabla\log\pi=\nabla\sum_x\pi=0$：k1 当 loss 完全没有有效梯度；
- $\mathbb E_\pi[\nabla k_3]=\sum_x(\pi-\pi_{ref})\nabla\log\pi=\nabla\,\text{KL}(\pi_{ref}\|\pi)$：值估计的是正向 KL，梯度却是**反向 KL** 的；
- $\mathbb E_\pi[\nabla k_2]=\sum_x\nabla\pi\,\log\frac{\pi}{\pi_{ref}}=\nabla\,\text{KL}(\pi\|\pi_{ref})$：正好是我们想要的梯度。

（测试 `test_expected_gradient_k1_k2_k3` 就是在整个 vocab 上按 $\pi$ 加权，把这三个期望精确算出来，再和 autograd 对真实 KL 求出的梯度对比。）

所以 verl 的 `kl_loss_type` 支持加 `+` 后缀（如 `k3+`）：**前向值用 k3，反向梯度用 k2**（straight-through），因为 k2 的梯度 $\nabla\frac12(\log r)^2 = -\log r\,\nabla\log\pi$ 恰好是 KL 梯度的无偏估计。

```python
# verl kl_penalty 的 "+" 写法（torch）
backward = 0.5 * (logprob - ref_logprob).square()
return backward - backward.detach() + forward.detach()
```

**放在哪里**：PPO-RLHF 常把 k1 作为逐 token 惩罚加进 reward（`use_kl_in_reward`）；GRPO 把 k3 直接加进 loss（`use_kl_loss`、`kl_loss_type=low_var_kl`）。

**易错点**

- k3 里是 $\ell_{ref}-\ell$（$\log r$），方向写反就变成了估计 $\text{KL}(\pi_{ref}\|\pi)$ 的另一个量，而且不再无偏。
- `exact_kl` 里 $p=0$ 的项直接写 `p * log(p / q)` 会得到 `0 * (-inf) = nan`。用 `torch.xlogy(p, p) - torch.xlogy(p, q)`（`xlogy(0, y) = 0`），或者 `torch.where(p > 0, ..., 0)`。

> [!Q]
> 为什么 k3 的 clamp 是必要的？
> 当 $\pi_{ref}\gg\pi$（策略把参考模型认为很可能的 token 压得很低）时 $e^{u}$ 会指数爆炸，一个 token 就能产生巨大的 loss 和梯度。clamp 到 ±10 牺牲一点无偏性换来数值稳定。

> [!Q]
> 用 KL 惩罚和不用（DAPO 去掉了 KL）各有什么理由？
> KL 惩罚防止策略离参考模型太远，减少 reward hacking 和语言退化；但在可验证奖励（数学、代码）的长思维链训练里，策略本来就需要大幅偏离初始模型，KL 会限制探索，所以 DAPO、Dr.GRPO 等去掉了它。
