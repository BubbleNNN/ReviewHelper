---
title: 集合通信：原语、Ring All-Reduce 与带宽
summary: 讲清每个通信原语谁发什么、发多少，用 α-β 模型估算通信时间，并知道它们分别出现在哪种并行策略里。
tags: NCCL, 通信, All-Reduce, 网络
order: 30
---

## 六个原语

设有 $p$ 个 rank，每个 rank 上有一块数据。下表的“每卡发送量”按带宽最优的实现计，$N$ 是完整张量的字节数。

| 原语 | 语义 | 每卡发送量 | 典型用途 |
|---|---|---|---|
| Broadcast | root 的数据复制给所有人 | 约 $N$ | 初始化参数、RL 权重同步 |
| Reduce | 所有人求和，结果只在 root | 约 $N$ | 汇总 loss/metric |
| All-Reduce | 所有人求和，人人拿到完整结果 | $2\frac{p-1}{p}N$ | DDP 梯度、TP 每层输出 |
| Reduce-Scatter | 求和后切成 $p$ 份，每人拿一份 | $\frac{p-1}{p}N$ | ZeRO-2/3、FSDP 梯度、Sequence Parallel |
| All-Gather | 每人贡献一份，人人拿到拼接结果 | $\frac{p-1}{p}N$ | FSDP 前向取参数、Sequence Parallel |
| All-to-All | 每人给每个其他人发不同的一块（分布式转置） | $\frac{p-1}{p}N$ | MoE 的 dispatch/combine、Ulysses |

还有点对点的 **Send/Recv**，用于 Pipeline Parallel 在相邻 stage 之间传激活和梯度。

> [!KEY]
> 记住一个恒等式：**All-Reduce = Reduce-Scatter + All-Gather**。ZeRO、FSDP、Sequence Parallel 都在利用这一点：把 all-reduce 拆开，中间只保留自己那 $1/p$，省显存而通信量不变。

## Ring All-Reduce

把 $p$ 张卡连成环，数据切成 $p$ 块：

1. **Reduce-Scatter 阶段**（$p-1$ 步）：第 $k$ 步，每个 rank 把手里某一块发给右邻居，同时收左邻居的一块并累加。$p-1$ 步后，每个 rank 恰好拥有某一块的完整和。
2. **All-Gather 阶段**（$p-1$ 步）：每个 rank 把自己那块完整结果沿环传下去，$p-1$ 步后人人都有全部 $p$ 块。

每步每卡发送 $N/p$ 字节，共 $2(p-1)$ 步，每卡总发送量 $2\frac{p-1}{p}N$。$p$ 很大时趋近 $2N$，**与卡数几乎无关**，这就是 ring 被称为带宽最优的原因。

<div class="flow"><span>GPU0</span><i>→</i><span>GPU1</span><i>→</i><span>GPU2</span><i>→</i><span>GPU3</span><i>↺</i><span class="f-infra">每步每卡发 N/p，共 2(p−1) 步</span></div>

动手：`exercises/system/ring_allreduce` 用 torch 张量在单进程里模拟这两个阶段，并统计步数和字节数。

## α-β 成本模型

一次发送 $n$ 字节的时间建模为 $T = \alpha + n/\beta$：$\alpha$ 是启动延迟（微秒级），$\beta$ 是带宽。

- Ring all-reduce：$T_{\text{ring}} = 2(p-1)\alpha + 2\frac{p-1}{p}\frac{N}{\beta}$。带宽项最优，但延迟项随 $p$ **线性**增长。
- Tree all-reduce（先 reduce 到根再 broadcast）：延迟项是 $O(\log p)\,\alpha$，但朴素二叉树的带宽项较差。NCCL 用 **double binary tree**，两棵互补的树各传一半数据，既有对数延迟又接近满带宽。

结论：**大消息用 ring（带宽主导），小消息、超多卡用 tree（延迟主导）**。NCCL 会根据消息大小和拓扑自动选算法（ring / tree / NVLS 等），可以用 `NCCL_ALGO` 强制。

> [!Q]
> **1024 卡做一次 4 字节的 all-reduce（比如同步一个 loss），用 ring 会怎样？**
> 带宽项可以忽略，延迟项是 $2 \times 1023 \times \alpha$。$\alpha$ 取 5 微秒，约 10 毫秒，对一个标量来说非常慢。tree 只要约 $2\log_2 1024 = 20$ 次延迟。所以小消息要么用 tree，要么攒起来一起发。

### busbw 与 algbw

`nccl-tests` 报两个数：**algbw** = 数据量 / 时间；**busbw** = algbw × 修正系数，修正后可以直接和链路带宽比较。all-reduce 的系数是 $2(p-1)/p$，all-gather 和 reduce-scatter 是 $(p-1)/p$。看到 busbw 接近链路带宽，说明通信实现已经跑满了硬件。

## 互联带宽的量级

| 链路 | 量级 | 备注 |
|---|---|---|
| NVLink 4（H100） | 每卡双向合计 900 GB/s，单向 450 GB/s | 节点内 8 卡经 NVSwitch 全互联 |
| NVLink 3（A100） | 每卡双向合计 600 GB/s | |
| NVLink 5（GB200 NVL72） | 每卡双向合计 1.8 TB/s | 72 卡在一个 NVLink 域 |
| InfiniBand NDR | 每端口 400 Gb/s = 50 GB/s | DGX H100 一般每卡配一张 400G 网卡 |
| RoCE v2 | 200/400 Gb/s | 以太网上的 RDMA，需要无损配置（PFC/ECN） |
| PCIe Gen5 x16 | 单向约 64 GB/s | CPU offload、无 NVLink 的卡间通信 |

节点内每卡单向约 450 GB/s，跨节点每卡约 50 GB/s，**相差近一个数量级**。这决定了并行策略的摆放：通信最密集的 TP 放节点内，DP、PP 可以跨节点。

其它要知道的词：

- **GPUDirect RDMA**：网卡直接读写 GPU 显存，不经过 CPU 内存中转。
- **NVLS / SHARP**：在 NVSwitch（NVLS）或 IB 交换机（SHARP）里做规约，all-reduce 的数据只需要“上去一次、下来一次”，进一步降低通信量。

## 估算练习：7B 模型 DDP 同步一次梯度要多久

7B 参数、bf16 梯度，$N = 14$ GB。按理想链路带宽估算（实测 busbw 会再低一些）：

- **单节点 8 卡，NVLink 单向 450 GB/s**：每卡发送 $2 \times \frac{7}{8} \times 14 \approx 24.5$ GB，约 $24.5/450 \approx 54$ ms。
- **2 个节点共 16 卡，扁平 ring**：环上必然有跨节点的边，瓶颈是每卡约 50 GB/s 的 IB。每卡发送 $2 \times \frac{15}{16} \times 14 \approx 26$ GB，约 0.5 秒，比单节点慢一个数量级。
- **分层（hierarchical）做法**：先在节点内 reduce-scatter（走 NVLink），每卡只剩 $1/8$ 的数据，再跨节点对这 $1/8$ 做 all-reduce（8 张卡的 IB 网卡并行），最后节点内 all-gather。跨节点部分每卡只需处理约 1.75 GB，NCCL 会根据拓扑自动选择类似的路径。

然后和一步的计算时间比较：如果反向本身要几百毫秒，54 ms 的通信完全可以被 bucket 重叠藏掉；0.5 秒就藏不住了，这时会考虑 ZeRO/HSDP 的分层策略、梯度压缩，或者加大每步的计算量（更大的 micro-batch、梯度累积）。

## 通信和计算重叠

通信放在独立的 CUDA stream 上，才能和计算并行：

- **DDP**：反向是从最后一层往前算梯度的，等一个 bucket 的梯度凑齐就立刻异步 all-reduce，同时继续算前面层的梯度。
- **FSDP**：在算第 $i$ 层时预取（prefetch）第 $i+1$ 层参数的 all-gather；反向时梯度的 reduce-scatter 和前一层的计算重叠。
- **TP**：all-reduce 在层内关键路径上，难重叠。Megatron 有把 GEMM 切块、和通信流水起来的 TP comm overlap 优化。
- **MoE**：all-to-all 通信量大。DeepSeek-V3 的 DualPipe 让一个 micro-batch 的计算和另一个的 all-to-all 重叠。

> [!WARN]
> NCCL kernel 本身也要占用 SM。通信和计算重叠时会抢 SM，所以“重叠”不是完全免费，实际加速比要实测。DeepEP 这类通信库会专门控制通信占用的 SM 数。

## DDP 的梯度 bucket

PyTorch DDP 不是每个参数单独 all-reduce，而是把梯度打包进 bucket（`bucket_cap_mb` 默认 25 MB）：

- bucket 按参数的**逆序**组织，大致对应反向时梯度产生的顺序；
- 一个 bucket 内所有梯度就绪后发起异步 all-reduce，和后续反向计算重叠；
- 太小的 bucket 浪费在延迟 $\alpha$ 上，太大的 bucket 要等很久才能开始发，重叠变差；
- 梯度累积时用 `no_sync()` 跳过中间 micro-step 的 all-reduce，只在最后一步同步。

> [!Q]
> **多卡训练卡住不动（hang），你怎么排查？**
> 最常见的原因是各 rank 调用集合通信的**顺序或次数不一致**，例如某个 rank 走了不同的 if 分支、某个 rank 的数据少一个 batch、某些参数在部分 rank 上没有梯度。排查：开 `NCCL_DEBUG=INFO` 看卡在哪个 op；设置合理的 timeout 让它报错而不是永远挂起；用 py-spy dump 各 rank 的栈对比；检查 dataloader 在各 rank 上的长度是否一致。

> [!Q]
> **为什么 MoE 的 all-to-all 比 dense 模型的 all-reduce 更难优化？**
> all-to-all 的数据量取决于路由结果，每次都不一样，负载可能不均（热门专家所在的卡收得多）；它又在前向关键路径上，每个 MoE 层前后各一次。跨节点时还受 IB 带宽限制。所以要配合负载均衡损失、限制每个 token 最多路由到几个节点（DeepSeek-V3 的 node-limited routing）、以及专门的通信库。
