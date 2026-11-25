---
title: Ring All-Reduce 模拟
difficulty: 中
tags: 通信, NCCL, 数据并行
order: 10
source: Baidu ring-allreduce / NCCL Ring 算法
---

用 numpy 模拟 $p$ 个 rank 上的 **ring all-reduce**：每个 rank 手里有一个长度 $N$ 的向量，结束时每个 rank 都要拿到所有向量的逐元素和。

规则：

- 每个 rank 把自己的向量用 `np.array_split(x, p)` 切成 $p$ 块，块编号 `0..p-1`（$N$ 不一定能被 $p$ 整除）。
- 环形拓扑：rank `r` **只能发给** rank `(r+1) % p`。
- 每一步（step）里**每个 rank 恰好发送一块**给下一个 rank，所有发送同时进行（先把要发的数据读出来，再统一写入）。
- 先做 $p-1$ 步 **reduce-scatter**（接收方把收到的块**加**到自己的同号块上），再做 $p-1$ 步 **all-gather**（接收方用收到的块**覆盖**自己的同号块）。step 编号从 0 连续到 $2(p-1)-1$。

```python
ring_allreduce(buffers) -> (results, log)
# buffers: 长度为 p 的 list，每个元素是 shape (N,) 的 np.ndarray
# results: 长度为 p 的 list，每个都是全体之和（shape (N,)）
# log:     list of (step, src, dst, chunk_id)，记录每一次发送
```

不要原地修改输入。

再实现一个 α-β 成本模型下的耗时估计（每条消息有固定延迟 $\alpha$，每字节耗时 $\beta$）：

```python
allreduce_time(n_bytes, p, alpha, beta) -> float
# ring all-reduce: 2(p-1) 步，每步发送 n_bytes / p 字节
```

<!-- solution -->

### 讲解

**思路。** 第 $s$ 步（reduce-scatter 阶段，$s=0..p-2$），rank $r$ 发送块 $(r-s) \bmod p$。经过 $p-1$ 步后，rank $r$ 手里的块 $(r+1) \bmod p$ 已经累加了全部 $p$ 份。all-gather 阶段第 $s$ 步，rank $r$ 发送块 $(r+1-s) \bmod p$，接收方直接覆盖。每一步里所有 rank 发出的块号互不相同，所以链路被完全并行地利用。

**通信量。** 每个 rank 一共发 $2(p-1)$ 块，每块 $N/p$，总共 $2\frac{p-1}{p}N$。当 $p$ 很大时趋于 $2N$，**和 rank 数几乎无关**，这就是 ring 被称为“带宽最优”的原因。代价是延迟项 $2(p-1)\alpha$ 随 $p$ 线性增长，小消息或超大规模时 NCCL 会改用 tree 算法。

$$T_{ring} = 2(p-1)\alpha + 2\frac{p-1}{p} N \beta$$

**易错点。**

- 同一步里必须“先读后写”，否则某个 rank 刚收到的数据会在同一步被转发出去，结果多加一遍。
- `np.array_split` 让前几块多一个元素，块大小不一定相等，按块号取切片即可。
- $p=1$ 时没有任何通信，直接返回输入的拷贝。

> [!Q]
> all-reduce 为什么可以拆成 reduce-scatter + all-gather？ZeRO / FSDP 为什么偏爱后者？
>
> 两者组合起来的通信量和 all-reduce 一样。ZeRO-2/3 和 FSDP 的梯度只需要 reduce-scatter 到各自的分片（每个 rank 只更新自己那 $1/p$ 参数），参数在用之前再 all-gather。把 all-reduce 拆开，正好对应“只存一片”的设计，显存省下来，通信量不变。

> [!Q]
> 8 卡 NVLink 和 2 机 16 卡 IB 上的 all-reduce，瓶颈分别在哪？
>
> 单机内 NVLink 带宽高，大消息时接近带宽上限。跨机时 ring 的每一步都要经过最慢的那条 IB 链路，所以 NCCL 做分层：先机内 reduce-scatter，再跨机 all-reduce，最后机内 all-gather。
