---
title: GPU 与 Roofline：算力、带宽和 MFU
summary: 用 roofline 判断一个算子是算力受限还是访存受限，并能当场估算训练 FLOPs、MFU 和 decode 吞吐上限。
tags: GPU, Roofline, MFU, 估算
order: 10
---

## GPU 的存储与计算层次

面试里讲 GPU，先把“数据在哪、算在哪”说清楚。以 H100 SXM 为例（数量级记住即可）：

| 层次 | 容量（H100 SXM） | 带宽量级 | 说明 |
|---|---|---|---|
| 寄存器 | 每 SM 256 KB | 最快 | 线程私有 |
| Shared memory / L1 | 每 SM 最多约 228 KB 可配为 shared | 十几 TB/s 级（片上 SRAM） | block 内线程共享，FlashAttention 的 tiling 就是往这里搬 |
| L2 cache | 50 MB | 数 TB/s 到十几 TB/s | 所有 SM 共享 |
| HBM3 | 80 GB | 约 3.35 TB/s | 权重、KV cache、激活都在这里 |
| NVLink 4 | — | 每卡双向合计 900 GB/s | 节点内卡间 |
| PCIe Gen5 x16 | — | 单向约 64 GB/s | CPU ↔ GPU，offload 的瓶颈 |

计算单元：H100 SXM 有 132 个 SM，每个 SM 有 4 个 Tensor Core 分区。Tensor Core 专门做矩阵乘累加（MMA），**bf16 dense 峰值约 989 TFLOPS**，FP8 dense 约 1979 TFLOPS；而普通 CUDA core 上的 FP32 非矩阵运算只有几十 TFLOPS（约 67 TFLOPS）。所以：

- 能变成 GEMM 的计算（Linear、Attention 的两次矩阵乘）跑在 Tensor Core 上，很快。
- 逐元素运算（激活函数、LayerNorm、softmax、残差加）几乎全是访存，瓶颈在 HBM。

对比参考：A100 SXM 80GB 是 bf16 dense 312 TFLOPS、HBM2e 约 2.0 TB/s、NVLink 3 双向 600 GB/s、108 个 SM、L2 40 MB。

> [!NOTE]
> 一句话概括 GPU 编程模型：**grid → block → warp（32 线程）→ thread**。一个 block 被调度到一个 SM 上，block 内线程共享 shared memory。kernel 优化的核心就是“让数据尽量在 SRAM/寄存器里复用，少回 HBM”。

## Roofline 模型

定义**算术强度**（arithmetic intensity）：

$$I = \frac{\text{FLOPs}}{\text{Bytes moved to/from HBM}}$$

一个 kernel 能达到的性能上限是

$$P = \min\left(P_{\text{peak}},\; I \times B_{\text{HBM}}\right)$$

两条线的交点叫 **ridge point**：$I^* = P_{\text{peak}} / B_{\text{HBM}}$。H100 上 $I^* \approx 989\text{e}12 / 3.35\text{e}12 \approx 295$ FLOPs/Byte；A100 约 $312/2.0 \approx 156$。

- $I < I^*$：**访存受限**（memory-bound），加算力没用，要减少数据搬运（融合 kernel、量化、复用）。
- $I > I^*$：**算力受限**（compute-bound），要提高 Tensor Core 利用率（对齐 shape、用更低精度）。

几个典型算子的算术强度（bf16，每个元素 2 字节）：

| 算子 | FLOPs | 字节 | 强度 | 结论 |
|---|---|---|---|---|
| 向量加 $y=x+z$，长度 $n$ | $n$ | $6n$ | 约 0.17 | 严重访存受限 |
| GEMM $(M\times K)\cdot(K\times N)$ | $2MKN$ | $2(MK+KN+MN)$ | $M=N=K=4096$ 时约 1365 | 算力受限 |
| GEMV（矩阵 × 向量，$M=1$） | $2KN$ | 约 $2KN$ | 约 1 | 访存受限，这就是 decode 慢的根源 |

> [!TIP]
> 估算 GEMM 强度的快捷方式：当 $M=1$（batch 为 1 的 decode），权重矩阵每个元素只被用一次，强度约等于 1 FLOP/Byte；当 batch 增大到 $B$，同一份权重被 $B$ 个 token 复用，强度近似变成 $B$。

## 训练 FLOPs：6ND

对一个参数量为 $N$ 的 dense Transformer，每个 token 的计算量近似：

- 前向：每个参数做一次乘加，$2N$ FLOPs；
- 反向：对激活的梯度 $2N$ + 对权重的梯度 $2N$，共 $4N$；
- 合计 **$6N$ FLOPs / token**，训练 $D$ 个 token 总计 $C \approx 6ND$。

如果开了 full activation recomputation，前向要多做一次，变成约 $8N$。注意 $6N$ 没算 attention 的 $QK^\top$ 和 $AV$ 两次矩阵乘，长序列时要加上：PaLM 附录给的每 token 计算量是 $6N + 12\,L\,d_{\text{model}}\,T$（$L$ 层，$T$ 为序列长度）。

推理时每生成一个 token 约 $2N$ FLOPs（只有前向）。

## MFU 怎么算

**MFU（Model FLOPs Utilization）** = 实际有用的模型 FLOPs ÷ 硬件峰值 FLOPs：

$$\text{MFU} = \frac{6N \cdot \text{tokens/s}}{P_{\text{peak}} \times \#\text{GPU}}$$

例：7B 模型在 H100 上每卡每秒训练 9000 个 token，则每卡有用算力 $6 \times 7\text{e}9 \times 9000 \approx 378$ TFLOPS，MFU $\approx 378/989 \approx 38\%$。大模型预训练 MFU 做到 40%–55% 算是不错。

> [!WARN]
> MFU 和 HFU（Hardware FLOPs Utilization）的区别：HFU 把重计算（recompute）的 FLOPs 也算进分子，所以 HFU ≥ MFU。论文报“利用率”时要看清是哪一个；MFU 只认“模型本身必须做的计算”，用于横向比较更公平。

> [!Q]
> **训练一个 70B 模型 2T token，用 1024 张 H100、MFU 40%，要多久？**
> 总计算 $6 \times 70\text{e}9 \times 2\text{e}12 = 8.4\text{e}23$ FLOPs；集群有效算力 $1024 \times 989\text{e}12 \times 0.4 \approx 4.05\text{e}17$ FLOPs/s；耗时约 $2.07\text{e}6$ 秒，约 24 天。

## Prefill 算力受限，Decode 访存受限

推理分两段：

<div class="flow"><span class="f-infra">Prefill<small>整段 prompt 一次前向，产出 KV cache</small></span><i>→</i><span class="f-accent">Decode<small>每步 1 个新 token，读全部权重 + KV</small></span><i>→</i><span>Decode …</span></div>

- **Prefill**：$T$ 个 token 一起过，每个权重被复用 $T$ 次，强度约为 $T$（几百到几千），远超 ridge point，**算力受限**。延迟指标是 TTFT。
- **Decode**：每步每条序列只有 1 个新 token。batch 为 $B$ 时权重强度约为 $B$，而 attention 部分还要读整个 KV cache，强度约为 1 且不随 batch 增大。小 batch 时**访存受限**。延迟指标是 TPOT。

### 数值例子：7B bf16 在 H100 上的 decode 上限

以 Llama-2-7B 为例（32 层，32 个头，head dim 128，MHA）：

1. **权重**：$7\text{e}9 \times 2 = 14$ GB。batch=1 时每生成一个 token 至少把权重从 HBM 读一遍。
2. **KV cache 每 token**：$2 \times 32 \times 32 \times 128 \times 2\ \text{B} = 0.5$ MB（K 和 V × 32 层 × 32 头 × head dim × bf16）。上下文 4096 时一条序列的 KV 为 2 GB。
3. **单条序列上限**：每步读 $14 + 2 = 16$ GB，$3.35\ \text{TB/s} \div 16\ \text{GB} \approx 210$ token/s；忽略 KV 时约 240 token/s。实测会更低（kernel 达不到满带宽、还有 launch 开销）。
4. **加大 batch**：权重读一次被 $B$ 条序列共享，权重部分强度约为 $B$。忽略 KV 时，要 $B \approx 295$ 才碰到算力上限。但 KV cache 随 $B$ 线性增长：$B=64$、每条 4096 上下文就要 128 GB KV，单卡放不下。**decode 吞吐最终被 KV cache 的容量和带宽卡住**，这正是 GQA/MLA、KV 量化、PagedAttention 要解决的问题。

> [!KEY]
> decode 的单 token 延迟下限 ≈ (权重字节 + 本步要读的 KV 字节) ÷ HBM 带宽。想提吞吐就加 batch，想降延迟就减少要读的字节（量化、GQA/MLA）或者一次多产出几个 token（投机解码）。

> [!Q]
> **为什么投机解码能加速 decode？**
> decode 是访存受限的，验证 $\gamma+1$ 个 token 和验证 1 个 token 读的权重量几乎一样，额外的计算“几乎免费”。但当 batch 很大、decode 已经接近算力受限时，这个前提就不成立了，见“投机解码”一篇。

## Kernel 层面的几个概念

写或读 CUDA/Triton kernel 时会反复遇到这几个词：

- **Warp 与 SIMT**：32 个线程组成一个 warp，同一时刻执行同一条指令。warp 内出现分支（`if` 走不同路径）时两条路径会串行执行，叫 **warp divergence**。
- **Occupancy**：一个 SM 上同时驻留的 warp 数占上限的比例。每个线程用的寄存器越多、每个 block 用的 shared memory 越多，能驻留的 warp 越少。访存受限的 kernel 需要较高的 occupancy，用大量 warp 轮流发起访存来掩盖 HBM 延迟（几百个时钟周期）。
- **Memory coalescing**：一个 warp 的 32 个线程访问连续地址时，硬件把它们合并成少数几次宽事务；跨步访问或随机访问会把带宽浪费掉。这就是为什么张量布局（行主序还是列主序、`contiguous()`）会影响速度。
- **Bank conflict**：shared memory 分成 32 个 bank，同一 warp 内多个线程访问同一 bank 的不同地址会被串行化，常用 padding 或 swizzle 布局避免。

## 怎么定位瓶颈

1. **先算理论值**：用上面的 roofline 估一下这个算子 / 这一步理论上该花多少时间，和实测比，差得越多越值得优化。
2. **看时间线**：Nsight Systems（`nsys`）或 PyTorch Profiler 看 kernel 时间线，找 GPU 空闲的空隙（CPU 瓶颈、同步、数据加载）和没被重叠的通信。
3. **看单个 kernel**：Nsight Compute（`ncu`）给出这个 kernel 达到的算力和带宽占峰值的比例，直接告诉你它离 roofline 的哪条线更近。
4. **常见结论**：小 batch decode 卡在 HBM 带宽；训练卡在通信没重叠；大量小算子卡在 launch 开销（用融合或 CUDA Graphs）。

> [!WARN]
> GPU 是异步执行的，用 Python 的 `time.time()` 包住一段 CUDA 代码只测到了 kernel 发射的时间。正确的计时要先 `torch.cuda.synchronize()`，或者用 CUDA event 记录起止点。

## 其它常被追问的点

- **为什么 shape 要对齐到 8/16/64 的倍数？** Tensor Core 的 MMA 指令按固定 tile 计算，维度不对齐会导致 padding 或回退到低效路径。vocab 常 pad 到 64/128 的倍数。
- **kernel launch 开销**：每次 launch 约几微秒。decode 阶段每步可能有几百个小 kernel，CPU 端调度会成为瓶颈，所以推理引擎用 CUDA Graphs 把一整步录下来回放。
- **Tensor Core 利用率低的常见原因**：矩阵太小（batch 小、TP 切得太细）、逐元素算子太多没有融合、通信没有和计算重叠。

> [!Q]
> **TP 切得越细越好吗？**
> 不是。TP 度为 $t$ 时，每卡 GEMM 的一个维度变成 $1/t$，矩阵变小、强度下降，同时每层要做 all-reduce。节点内 TP=8 已经是常见上限，再往上收益为负。

动手：`exercises/system/memory_estimator`（算显存与 KV）、`exercises/model/attention`。
