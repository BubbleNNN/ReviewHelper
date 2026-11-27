---
title: AdamW 一步更新
difficulty: 易
tags: 优化器, 显存
order: 36
source: Kingma & Ba 2014 Adam；Loshchilov & Hutter 2017 AdamW；torch.optim.AdamW
---

实现和 `torch.optim.AdamW` 一致的单步更新。`param`、`grad` 是同形状的 `torch.Tensor`（测试用 float64）：

```python
adamw_step(param, grad, state, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0) -> new_param
```

- `state` 是一个 dict，第一次调用时为空 `{}`，由你初始化为 `{"step": 0, "m": zeros, "v": zeros}`，并**原地更新**。`step` 是 python int，`m`、`v` 是和 `param` 同形状、同 dtype 的 tensor。
- 返回新的参数 tensor（可以新建，不要求原地修改 `param`），dtype、device 与 `param` 一致。
- 不要调用 `torch.optim.AdamW` / `torch.optim.Adam` 或 `torch.optim` 里的其他优化器：测试拿它们当参照答案逐步对拍。

每一步（$t$ 从 1 开始）：

$$\theta \leftarrow \theta(1-\eta\lambda)$$

$$m \leftarrow \beta_1 m + (1-\beta_1)g,\quad v\leftarrow\beta_2 v + (1-\beta_2)g^2$$

$$\hat m = \frac{m}{1-\beta_1^t},\quad \hat v=\frac{v}{1-\beta_2^t},\quad \theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

<!-- solution -->

### 讲解

**偏差修正**：$m,v$ 从 0 开始，前几步被严重低估（第 1 步 $m=0.1g$）。除以 $1-\beta^t$ 后第 1 步 $\hat m=g$、$\hat v=g^2$，更新量约为 $\eta\cdot\text{sign}(g)$。这也说明 Adam 的步长**和梯度尺度无关**（梯度整体乘 1000，更新几乎不变），所以对 loss 缩放不敏感。

**AdamW vs Adam + L2**：Adam 里如果把 $\lambda\theta$ 加进梯度，它会被 $\sqrt{\hat v}$ 除掉，梯度大的参数实际衰减反而小。AdamW 把权重衰减和梯度更新解耦，直接 $\theta \leftarrow \theta(1-\eta\lambda)$，和 SGD 的权重衰减语义一致。

**显存账**：混合精度 Adam 每个参数要存 fp32 主权重（4B）+ m（4B）+ v（4B），再加 bf16 权重（2B）和梯度（2B），合计约 16 字节/参数。7B 模型光优化器相关状态就是约 112GB，这就是 ZeRO / FSDP 要切分优化器状态的原因（见 Infra 理论的显存账）。

**易错点**

- `step` 先加 1 再算偏差修正。
- $\epsilon$ 加在 $\sqrt{\hat v}$ 外面（PyTorch 的写法），不是加在根号里面。
- 权重衰减先于梯度更新施加（PyTorch AdamW 的顺序），这会带来 $O(\eta^2)$ 级别的差异。

> [!Q]
> 为什么 LLM 训练常用 β2=0.95 而不是 0.999？
> β2 越大，$v$ 对梯度突变的反应越慢。大模型训练中偶发的梯度尖峰会让旧的小 $v$ 来不及跟上，导致更新量过大、loss 飙升。0.95 让 $v$ 更快适应，训练更稳定（GPT-3、Llama 都用 0.95）。

```python
# 实际训练时直接用 torch 自带的（注意它的默认 weight_decay=0.01，不是 0）
opt = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01)
```
