---
title: Megatron 张量并行 MLP（前向 + 反向）
difficulty: 难
tags: 张量并行, Megatron, 反向传播
order: 12
source: Megatron-LM (Shoeybi et al., 2019) · ColumnParallelLinear / RowParallelLinear
---

在单进程里用 numpy 模拟 `tp` 个 rank 上的 Megatron 风格 MLP：

$$Y = \mathrm{GeLU}(X W_1 + b_1)\, W_2 + b_2,\quad X\in\mathbb{R}^{n\times d},\ W_1\in\mathbb{R}^{d\times h},\ W_2\in\mathbb{R}^{h\times d}$$

切分方式：

- $W_1$、$b_1$ **按列**切（column parallel，按隐藏维 $h$ 切），每个 rank 独立算 $Z_r = X W_1^{(r)} + b_1^{(r)}$ 和 $A_r=\mathrm{GeLU}(Z_r)$。GeLU 是逐元素的，所以**不需要通信**。
- $W_2$ **按行**切（row parallel，同样按 $h$ 切），每个 rank 算部分和 $Y_r = A_r W_2^{(r)}$，最后 **all-reduce 求和**，再加 $b_2$（只加一次）。
- 用 `np.array_split` 切分，$h$ 不一定能被 `tp` 整除。

```python
shard_weights(W1, b1, W2, tp) -> list[dict]           # 每个 rank: {"W1", "b1", "W2"}
tp_mlp_forward(x, shards, b2, all_reduce=None) -> (y, cache)
tp_mlp_backward(dy, x, shards, cache, all_reduce=None) -> grads
# grads = {"dx": ..., "db2": ..., "dW1": [每 rank], "db1": [每 rank], "dW2": [每 rank]}
```

`all_reduce(list_of_arrays) -> array` 是“通信原语”，默认实现为逐元素求和。**前向和反向都只允许调用它一次**。测试会传入一个计数的 `all_reduce` 来检查。

这对应 Megatron 论文里的两个共轭算子：

| 算子 | 位置 | 前向 | 反向 |
|---|---|---|---|
| $f$ | column-parallel 的输入 $X$ | identity | all-reduce（把各 rank 的 $\partial L/\partial X$ 相加） |
| $g$ | row-parallel 的输出 $Y$ | all-reduce | identity |

模板里已经给了 `gelu(x)` 和 `gelu_grad(x)`（tanh 近似），直接用即可。

<!-- solution -->

### 讲解

**为什么这样切？** 第一层按列切，每个 rank 拿到完整的 $X$，产出隐藏层的一段列，GeLU 逐元素作用在这段上，结果和先拼起来再算一样。第二层按行切，恰好吃进本 rank 那段隐藏层，输出是完整形状的**部分和**，一次 all-reduce 就得到真值。整个 MLP 前向只通信一次，反向也只一次（$f$ 处对 $dX$ 求和）。如果第一层按行切，GeLU 之前就要先 all-reduce，否则非线性作用在部分和上是错的。

**反向推导。** 设 $dY$ 已知：

- $dW_2^{(r)} = A_r^\top dY$，$db_2 = \sum_i dY_i$（$g$ 的反向是 identity，各 rank 拿到相同的 $dY$）
- $dA_r = dY\, W_2^{(r)\top}$，$dZ_r = dA_r \odot \mathrm{GeLU}'(Z_r)$
- $dW_1^{(r)} = X^\top dZ_r$，$db_1^{(r)} = \sum_i dZ_{r,i}$
- $dX = \sum_r dZ_r W_1^{(r)\top}$，**这里需要 all-reduce**

**通信量。** 每层 Transformer 的 attention 和 MLP 各要 2 次 all-reduce（前向 1 次 + 反向 1 次），每次大小 $n\times d$。TP 的通信在关键路径上、频繁、量大，所以一般限制在一台机器的 NVLink 域内（TP≤8）。

**易错点。**

- $b_2$ 只能在 all-reduce 之后加一次，每个 rank 都加就会变成 $tp \cdot b_2$。
- $b_1$ 要跟 $W_1$ 一起按列切。
- 反向里 $dX$ 要求和，但 $dW$ 不需要通信：每个 rank 只更新自己那片权重。

> [!Q]
> Megatron 的 sequence parallel 又改了什么？
>
> TP 区域外的 LayerNorm、Dropout 在每个 rank 上是重复计算、重复存激活的。SP 把这些部分沿序列维切开，并把 all-reduce 拆成 reduce-scatter + all-gather。通信量不变，激活显存再省约 $1/tp$。

> [!Q]
> 为什么 attention 的 TP 按 head 切很自然？
>
> 各个 head 之间独立。QKV 投影按列切（每个 rank 负责若干 head），输出投影按行切，和 MLP 是同一个模式。所以要求 head 数能被 tp 整除，GQA 时 kv head 数也要考虑。
