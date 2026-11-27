---
title: Scaled Dot-Product Attention（causal + padding mask）
difficulty: 易
tags: Transformer, Attention, mask
order: 14
source: Vaswani et al. 2017；torch.nn.functional.scaled_dot_product_attention
---

实现带 mask 的缩放点积注意力：

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right)V$$

```python
scaled_dot_product_attention(q, k, v, causal=False, key_padding_mask=None) -> (out, attn)
```

- 输入输出都是 `torch.Tensor`（测试用 float64），输出 dtype 与 `q` 相同。
- `q`: `(B, H, Tq, d)`，`k`: `(B, H, Tk, d)`，`v`: `(B, H, Tk, dv)`
- `key_padding_mask`: `(B, Tk)` 的 bool tensor，**True 表示有效 token**，False 表示 padding（要被 mask 掉）。可以为 `None`。
- `causal=True` 时用**右下角对齐**：第 `i` 个 query 只能看第 `j <= i + (Tk - Tq)` 个 key。`Tq == Tk` 时就是普通的下三角；`Tq=1` 时（带 KV cache 解码）能看到全部 key。
- 返回 `out`: `(B, H, Tq, dv)` 和注意力权重 `attn`: `(B, H, Tq, Tk)`，被 mask 的位置权重必须严格为 0。
- 保证每个 query 至少有一个可见的 key。
- **不要直接调用** `F.scaled_dot_product_attention`、`nn.MultiheadAttention` / `F.multi_head_attention_forward`（测试拿 `F.scaled_dot_product_attention` 当对照）。`torch.softmax` 可以用。

<!-- solution -->

### 讲解

**步骤**：算分数 `q @ k^T / sqrt(d)` → 把不允许的位置设成 `-inf` → 沿最后一维做稳定 softmax → 乘 `v`。

**为什么除以 $\sqrt d$**：如果 $q,k$ 的分量独立、均值 0、方差 1，点积的方差是 $d$。不缩放的话 $d$ 一大，softmax 的输入幅度很大，输出接近 one-hot，梯度几乎为 0。

**两种 mask 的形状**

- causal mask 只和位置有关：`(Tq, Tk)`，广播到所有 batch 和 head。
- padding mask 只和 key 有关：`(B, Tk)` → `(B, 1, 1, Tk)`。
- 两者取交集（都允许才可见）。

**复杂度**：时间 $O(B H T_q T_k d)$，显存要存 $O(BHT_qT_k)$ 的分数矩阵，这正是长序列的瓶颈，也是 FlashAttention 要解决的问题。

**易错点**

- padding 方向：左 padding（生成时常用）还是右 padding，mask 都按 key 维度来，不要在 query 维度上 mask 掉再忘了处理整行 `-inf`。
- 右下角对齐和左上角对齐只在 `Tq != Tk` 时有区别。PyTorch 的 `is_causal=True` 用的是左上角对齐，KV cache 场景要自己构造 mask。

> [!Q]
> 如果某个 query 的所有 key 都被 mask 了会怎样？
> softmax 的输入整行是 `-inf`，结果是 `nan`，并且会通过反向传播污染整个 batch。常见处理：保证至少有一个可见 key（比如让 padding 的 query 看自己），或者用 `-1e4` / `finfo.min` 代替 `-inf`，再把这些行的输出置零。

```python
# torch
scores = q @ k.transpose(-1, -2) / math.sqrt(d)
scores = scores.masked_fill(~allowed, float("-inf"))
out = scores.softmax(-1) @ v
```
