---
title: FlashAttention 前向（分块 + online softmax）
difficulty: 难
tags: Attention, Kernel, online softmax
order: 20
source: Dao et al. 2022 FlashAttention；Milakov & Gimelshein 2018 Online softmax
---

用**分块 + online softmax** 实现单头注意力前向，不允许一次性构造完整的 `(Tq, Tk)` 分数矩阵（每次只处理一个 `(block_q, block_k)` 的块）。

```python
flash_attention(q, k, v, causal=False, block_q=16, block_k=16) -> (out, lse)
```

- `q`: `(Tq, d)`，`k`: `(Tk, d)`，`v`: `(Tk, dv)`；分数要乘 `1/sqrt(d)`。
- `causal=True` 时保证 `Tq == Tk`，query `i` 只看 key `j <= i`。
- `out`: `(Tq, dv)`，必须和朴素实现数值一致。
- `lse`: `(Tq,)`，每行分数的 logsumexp（FlashAttention 反向要用它重算 softmax）。
- 长度不一定能被块大小整除。

对每个 query 块，维护三个量：行最大值 $m$、归一化分母 $\ell$、未归一化的输出累加 $O$。处理新的 key 块（分数 $S$）时：

$$m' = \max(m, \text{rowmax}(S)),\quad P = e^{S - m'},\quad \ell' = e^{m-m'}\ell + \text{rowsum}(P),\quad O' = e^{m-m'}O + PV$$

最后 `out = O / ℓ`，`lse = m + log ℓ`。

<!-- solution -->

### 讲解

**为什么要分块**：朴素实现要把 $T\times T$ 的分数矩阵写回 HBM 再读出来做 softmax，HBM 读写量是 $O(T^2)$；而 GPU 上 attention 主要受限于访存，不是算力。FlashAttention 把 $Q,K,V$ 切成能放进 SRAM（共享内存）的块，在片上完成 `matmul → softmax → matmul`，分数矩阵从不落 HBM。

**IO 复杂度**：设 SRAM 大小为 $M$，FlashAttention 的 HBM 访问量是 $O(T^2 d^2 / M)$，标准实现是 $O(Td + T^2)$。$d$ 通常 64–128，$M$ 约 100KB 量级，所以 $d^2/M \ll 1$，访存降很多。FLOPs 不变（反向还会多一次重算），但 wall-clock 显著变快，显存从 $O(T^2)$ 降到 $O(T)$。

**online softmax 的正确性**：新块进来后最大值变大，旧的 $\ell$ 和 $O$ 都是在旧最大值下累加的，乘上 $e^{m-m'}$ 就换算到了新基准。所有块处理完后，$O/\ell$ 就等于完整 softmax 的结果。

**causal 的两个优化**：key 块起点大于 query 块终点时整块跳过（省一半计算）；只有对角块需要逐元素 mask。

**易错点**

- 初始 $m=-\infty$。第一次更新时 $e^{m-m'}=e^{-\infty}=0$，不要写成 `exp(-inf - (-inf))`。
- 某个块里一整行都被 mask 时 rowmax 是 `-inf`，要防止出现 `nan`。
- 除以 $\ell$ 放在最后做一次，而不是每块都归一化（FA2 的改进之一，减少非 matmul 的 FLOPs）。

> [!Q]
> FlashAttention 的反向为什么不存注意力矩阵？
> 只存每行的 $\text{lse}$（$O(T)$），反向时用 $P=\exp(S-\text{lse})$ 重新算出注意力概率。重算多花一些 FLOPs，但省掉了 $O(T^2)$ 的显存和 HBM 读写，整体更快。

> [!Q]
> FA2、FA3 分别改了什么？
> FA2：减少非 matmul 运算；在序列维度上并行（不只 batch×head）；调整 warp 间的分工，减少共享内存通信。FA3（Hopper）：用 TMA/WGMMA 做异步数据搬运和矩阵乘，warp specialization 让 softmax 和 GEMM 重叠，并支持 FP8。
