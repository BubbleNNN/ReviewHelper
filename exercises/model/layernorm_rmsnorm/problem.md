---
title: LayerNorm 前向/反向与 RMSNorm
difficulty: 中
tags: 归一化, 反向传播
order: 12
source: Ba et al. 2016 LayerNorm；Zhang & Sennrich 2019 RMSNorm（Llama 系列使用）
---

输入 `x` 是形状 `(N, D)` 的 `torch.Tensor`（测试用 float64），在最后一维做归一化。输出 dtype 与输入一致。

**LayerNorm**（方差用有偏估计，即除以 $D$）：

$$\mu=\frac1D\sum_j x_j,\quad \sigma^2=\frac1D\sum_j (x_j-\mu)^2,\quad \hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y=\gamma\odot\hat x+\beta$$

**RMSNorm**（不减均值，没有 $\beta$）：

$$y = \frac{x}{\sqrt{\frac1D\sum_j x_j^2+\epsilon}}\odot w$$

```python
layernorm_forward(x, gamma, beta, eps=1e-5) -> (y, cache)   # cache 内容自定，反向时用
layernorm_backward(dy, cache) -> (dx, dgamma, dbeta)         # dgamma/dbeta 形状 (D,)
rmsnorm(x, weight, eps=1e-6) -> y
```

`layernorm_backward` 要给出对 `x`、`gamma`、`beta` 的梯度。测试会用有限差分和 autograd（对 `F.layer_norm` 求导）两种方式检查。

**不许调库**：不要调用 `F.layer_norm`、`F.rms_norm`、`torch.nn.LayerNorm` / `RMSNorm`；反向要手推公式实现，**不能用 autograd**（`.backward()`、`torch.autograd.grad` 都不行），测试会在 `torch.no_grad()` 下调用你的前向和反向。

<!-- solution -->

### 讲解

**前向**很直接。**反向**的关键是 $\hat x$ 同时依赖每个 $x_j$（通过 $\mu$ 和 $\sigma$）。记 $g=\gamma\odot dy$（对 $\hat x$ 的梯度），$r=1/\sqrt{\sigma^2+\epsilon}$，可以推出：

$$dx = r\left(g-\overline{g}-\hat x\cdot\overline{g\odot\hat x}\right)$$

其中上划线表示沿 $D$ 维取均值。$d\gamma=\sum_n dy\odot\hat x$，$d\beta=\sum_n dy$。

推导思路：$\hat x$ 关于 $x$ 的雅可比是 $r\,(I-\frac1D\mathbf 1\mathbf 1^\top-\frac1D\hat x\hat x^\top)$，它是对称的，所以 $dx$ 等于这个矩阵乘 $g$。

**两个可以拿来自检的性质**

- LayerNorm 对平移不变：$\text{LN}(x+c\mathbf 1)=\text{LN}(x)$，所以 $\sum_j dx_j=0$。
- RMSNorm 对正缩放不变（$\epsilon=0$ 时）：$\text{RMS}(cx)=\text{RMS}(x)$。

**易错点**

- 方差用有偏估计（除以 $D$）。`torch.var` 默认是无偏的（除以 $D-1$），手写 LayerNorm 时要传 `correction=0`（旧写法 `unbiased=False`），或者直接写 `((x - mu) ** 2).mean(-1)`。
- 混合精度训练里归一化要在 fp32 里算，再 cast 回 bf16，否则 $\sigma^2$ 精度不够。

> [!Q]
> 为什么 Llama 用 RMSNorm 而不是 LayerNorm？
> RMSNorm 少一次减均值、少一个 bias，计算和访存都更省。实验上效果相当：去掉“重新居中”这一步几乎不掉点，“重新缩放”才是起作用的部分。

> [!Q]
> Pre-LN 和 Post-LN 有什么区别？
> Post-LN（原始 Transformer）在残差相加之后做归一化，深层时梯度不稳定，需要 warmup。Pre-LN 在子层之前归一化，残差通路是恒等映射，训练更稳，现代 LLM 基本都用 Pre-LN（或 Pre-RMSNorm）。

```python
# 工程里 RMSNorm 一般这样写：bf16/fp16 输入先升 fp32 再算，最后 cast 回去
# （本题测试用 float64，不需要这一步；x.float() 反而会把 float64 降成 float32）
def rmsnorm(x, w, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w
```
