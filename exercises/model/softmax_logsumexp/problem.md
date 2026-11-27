---
title: 数值稳定的 softmax / log_softmax / logsumexp
difficulty: 易
tags: 数值稳定, 基础算子
order: 10
source: 所有 LLM 实现的第一行；torch.logsumexp / F.log_softmax
---

实现三个沿 `dim` 计算的函数，输入是任意形状的 `torch.Tensor`（测试用 float64），输出 dtype 与输入相同：

```python
logsumexp(x, dim=-1, keepdim=False)   # log Σ_j exp(x_j)
softmax(x, dim=-1)                     # exp(x_i) / Σ_j exp(x_j)
log_softmax(x, dim=-1)                 # x_i - logsumexp(x)
```

要求：

- **不要直接调用** `torch.logsumexp`、`torch.softmax` / `F.softmax`、`torch.log_softmax` / `F.log_softmax`（测试拿它们当对照）。用 `torch.amax`、`torch.exp`、`torch.log`、`.sum`、`torch.where` 这些基础算子自己写。
- 输入里有 `1e4` 这样的大 logit 时不能溢出成 `inf/nan`；`-1000` 这样的小 logit 不能下溢成 `log(0)`。
- 支持 `-inf`（被 mask 的位置）：softmax 给 0，log_softmax 给 `-inf`。
- **整行都是 `-inf` 时，`logsumexp` 返回 `-inf`**（不能是 `nan`）。softmax 在这种行上没有定义，不测。
- `logsumexp` 的 `keepdim` 语义和 `torch.logsumexp` 一致；一维输入、`keepdim=False` 时返回 0-d tensor。

<!-- solution -->

### 讲解

**核心恒等式**：对任意常数 $c$，

$$\log\sum_j e^{x_j} = c + \log\sum_j e^{x_j - c}$$

取 $c=\max_j x_j$ 后每一项 $e^{x_j-c}\le 1$，不会上溢；而且最大那一项恰好是 $e^0=1$，求和至少为 1，`log` 不会碰到 0，也就不会下溢。softmax 同理：分子分母同乘 $e^{-c}$。

**易错点**

- 整行 `-inf` 时 $c=-\infty$，`x - c` 变成 `-inf - (-inf) = nan`。解决办法是把非有限的 max 换成 0：`m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))`。
- `x.max(dim=d, keepdim=True)` 返回的是 `(values, indices)` 二元组，要取 `.values`；或者直接用 `torch.amax(x, dim=d, keepdim=True)`。
- `log_softmax` 不要写成 `log(softmax(x))`：当某个概率小于 `1e-308` 时 softmax 下溢为 0，再取 log 得到 `-inf`，而 `x - logsumexp(x)` 能给出准确的有限值。PPO / GRPO 里的 logprob 一律用这种写法。
- 用 `-1e9` 还是 `-inf` 做 mask：`-inf` 更干净，但要保证每行至少有一个有效位置；`-1e9` 在 fp16 下会溢出（fp16 最大约 65504），所以混合精度代码里常用 `torch.finfo(dtype).min`。

**复杂度**：$O(N)$，需要两遍扫描（先求 max，再求和）。online softmax 能把它合成一遍，这正是 FlashAttention 的出发点，见 `exercises/model/flash_attention`。

> [!Q]
> bf16 下直接算 logsumexp 有什么问题？
> bf16 只有 8 位尾数（7 位显式存储，相对精度约 $4\times10^{-3}$），对很长的 vocab 求和时累加误差很大。实际实现会把 logits 升到 fp32 再做 `logsumexp`，这也是 verl 里算 logprob 前通常先 `logits.float()` 的原因。

```python
# torch 一行版
logp = torch.log_softmax(logits.float(), dim=-1)
lse = torch.logsumexp(logits.float(), dim=-1)
```
