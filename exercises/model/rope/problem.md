---
title: RoPE 旋转位置编码
difficulty: 中
tags: Transformer, 位置编码
order: 18
source: Su et al. 2021 RoFormer；HF transformers 的 rotate_half / apply_rotary_pos_emb
---

实现 RoPE，采用 **HF / Llama 的 `rotate_half` 约定**：

- 频率：$\theta_i = \text{base}^{-2i/d}$，$i = 0,\dots,d/2-1$。
- 把向量**前一半和后一半**配对：第 $i$ 维和第 $i + d/2$ 维组成一对，用角度 $p\,\theta_i$ 旋转（$p$ 是位置）。
- `cos`/`sin` 表写成 `concat([pθ, pθ])`，长度为 $d$；`rotate_half(x) = concat([-x2, x1])`，其中 `x1, x2` 是前后两半。

$$\text{RoPE}(x, p) = x \odot \cos(p\theta) + \text{rotate\_half}(x) \odot \sin(p\theta)$$

```python
rope_cos_sin(positions, dim, base=10000.0) -> (cos, sin)   # 各 (T, dim)，float64
rotate_half(x)                                               # (..., dim)
apply_rope(x, positions, base=10000.0)                       # x: (..., T, dim)，positions: (T,)
```

输入输出都是 `torch.Tensor`：`positions` 是 long tensor，`x` 是浮点 tensor（测试用 float64）。`dim` 一定是偶数。

- `rope_cos_sin` 一律用 float64 计算并返回 float64 的表（位置大时角度 $p\theta$ 很大，float32 精度不够）。
- `apply_rope` 的输出 dtype 与 `x` 相同；要能处理 `(B, H, T, dim)` 这样的多维输入（在最后两维上广播）。

<!-- solution -->

### 讲解

**为什么旋转能编码相对位置**：把第 $i$ 对看成复数 $z_i$，RoPE 就是 $z_i \mapsto z_i e^{\mathrm{i} p\theta_i}$。两个向量点积的实部

$$\langle \text{RoPE}(q,m), \text{RoPE}(k,n)\rangle = \text{Re}\sum_i q_i \bar k_i e^{\mathrm{i}(m-n)\theta_i}$$

只依赖 $m-n$。所以注意力分数天然是相对位置的函数，而且旋转不改变向量范数。

**两种实现约定**

| 约定 | 配对方式 | 谁在用 |
|---|---|---|
| interleaved | $(x_0,x_1),(x_2,x_3),\dots$ | 原论文、Meta 原版 Llama（复数实现） |
| rotate_half | $(x_0,x_{d/2}),(x_1,x_{d/2+1}),\dots$ | HF transformers、多数推理框架 |

两者数学上等价（相当于对 $q,k$ 的维度做了同一个置换，点积不变），但**权重不能混用**：HF 转换 Meta 原版权重时会对 `wq`/`wk` 做一次维度重排。

**易错点**

- 只对 $q,k$ 加 RoPE，不加在 $v$ 上。
- KV cache 里存的是**已经旋转过的 k**；解码时新 token 的位置是 `T_cached`，不是 0。
- 频率指数是 $-2i/d$，即 `torch.arange(0, dim, 2, dtype=torch.float64) / dim`。不写 dtype 的话 `arange` 是 int64，除以 `dim` 后变成默认的 float32，精度不够。

**长度外推**：base 越大，低频分量周期越长。NTK-aware scaling 通过增大 base 来外推；YaRN 对不同频段分别插值并调整注意力温度。

> [!Q]
> 为什么 RoPE 比绝对位置编码更适合长上下文？
> 相对位置信息直接体现在 $q\cdot k$ 里，模型不需要从加性的位置向量里“解码”距离；而且随着距离增大，高频分量的贡献会互相抵消，带来一定的远程衰减。外推仍然受训练长度限制，所以才需要 NTK / YaRN 这类缩放。

```python
# torch（HF 写法）
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
q_embed = q * cos + rotate_half(q) * sin
```
