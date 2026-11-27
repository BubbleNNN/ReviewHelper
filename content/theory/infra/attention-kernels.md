---
title: Attention 与算子：FlashAttention、GQA/MLA、RoPE、低精度
summary: 从 IO 角度理解 FlashAttention，比较各种 KV 共享方案对 KV cache 的影响，并讲清 RoPE 和 bf16/fp16/fp8 的取舍。
tags: FlashAttention, GQA, MLA, RoPE, 混合精度
order: 50
---

## 标准 Attention 慢在哪

$$O = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V,\quad Q,K,V \in \mathbb{R}^{N\times d}$$

朴素实现分三步：算 $S=QK^\top$ 写回 HBM → 读 $S$ 算 softmax 得 $P$ 写回 HBM → 读 $P$ 算 $PV$。中间矩阵是 $N \times N$ 的，HBM 访问量是 $\Theta(Nd + N^2)$，显存也是 $O(N^2)$。$N=8192$ 时单个头的 $S$ 就有 6700 万个元素。而 softmax 这步本身是访存受限的，于是 attention 的大部分时间花在搬运 $N^2$ 矩阵上，而不是计算上。

## FlashAttention：tiling + online softmax

核心思想：**不把 $N\times N$ 矩阵写回 HBM**。把 $Q$ 切成行块、$K,V$ 切成列块，每次把一个 $Q$ 块和一个 $K,V$ 块搬进 SRAM 算完，只维护三个小量：

- $m$：当前见过的每行最大值；
- $\ell$：当前每行的 $\sum e^{s - m}$；
- $O$：未归一化的输出累加。

处理新的一块 $S_j = Q_i K_j^\top$ 时：

$$m' = \max(m, \text{rowmax}(S_j)),\qquad \ell' = e^{m-m'}\ell + \text{rowsum}(e^{S_j - m'})$$

$$O' = e^{m-m'}\,O + e^{S_j - m'}\,V_j$$

所有块处理完后输出 $O/\ell$。旧的累加量乘 $e^{m-m'}$ 是为了把它们“换算”到新的最大值下，这就是 **online softmax**，结果和一次性 softmax 精确相等（不是近似）。

- **IO 复杂度**：HBM 访问 $\Theta(N^2 d^2 / M)$，$M$ 是 SRAM 大小。$d$ 通常为 64–128，$M$ 约 100 KB 量级，比朴素实现少很多倍。
- **显存**：只额外存每行的 logsumexp $L = m + \log \ell$，$O(N)$。
- **反向**：不存 $P$，用 $Q, K$ 和 $L$ 重新算出 $P$。FLOPs 变多了，但 IO 少了，整体反而更快。

> [!KEY]
> FlashAttention 的 FLOPs 一点没少，甚至反向更多。它快是因为把一个访存受限的算子变成了接近算力受限的算子。面试时一定要从 IO 讲，而不是说“它优化了计算”。

动手：`exercises/model/flash_attention` 用 torch 实现分块 + online softmax，并验证和朴素 attention 数值一致。

### FA2 与 FA3 改进了什么

- **FlashAttention-2**：①减少非矩阵乘的 FLOPs（不在每一步都对 $O$ 做缩放，最后统一除）；②除了 batch 和 head，还在**序列维**上并行，长序列小 batch 时 SM 利用率更高；③block 内把 $Q$ 分给不同 warp（而不是把 $K$ 分给不同 warp），减少 warp 间通过 shared memory 交换数据。A100 上约为 FA1 的 2 倍。
- **FlashAttention-3**（Hopper）：用 warp specialization + TMA 让数据搬运和计算异步进行；两组 warpgroup ping-pong 调度，让一组的 softmax 和另一组的 GEMM 重叠；支持 FP8（分块量化 + Hadamard 变换降低离群值误差）。论文报告 H100 上 BF16 最高约 740 TFLOPS（约 75% 利用率），FP8 接近 1.2 PFLOPS（数字待核实）。
- **Flash-Decoding**：decode 时 query 只有 1 行，batch × head 可能填不满 SM，于是沿 KV 长度再切分并行，最后用 logsumexp 合并。

## MHA / MQA / GQA / MLA

decode 时要把所有历史 K、V 从 HBM 读一遍，KV 越小越快、能并发的请求越多。

| 方案 | 每层每 token 缓存的元素数 | 质量 | 说明 |
|---|---|---|---|
| MHA | $2\,n_h\,d_h$ | 基准 | 每个 query 头有自己的 K、V |
| MQA | $2\,d_h$ | 有下降 | 所有 query 头共享一组 K、V |
| GQA（$g$ 组） | $2\,g\,d_h$ | 接近 MHA | 每 $n_h/g$ 个 query 头共享一组；Llama-2-70B 用 $g=8$ |
| MLA | $d_c + d_h^R$ | DeepSeek-V2 报告不差于 MHA | 缓存低秩 latent，DeepSeek-V2/V3 为 $512+64=576$ |

GQA 还有一个附带好处：同一份 K、V 被 $n_h/g$ 个 query 头复用，decode attention 的算术强度提高了 $n_h/g$ 倍。

**MLA 的两个关键点**：

1. **矩阵吸收**：K、V 由 latent $c_{KV}$ 上投影得到（$k = W^{UK} c_{KV}$）。推理时可以把 $W^{UK}$ 并进 $W^Q$、把 $W^{UV}$ 并进输出投影，直接在 latent 空间算 attention，不需要还原出每个头的完整 K、V。
2. **解耦 RoPE**：RoPE 的旋转矩阵夹在 $q$ 和 $k$ 中间，依赖位置，会破坏上面的吸收。所以 MLA 额外用一小段维度（$d_h^R=64$）单独承载 RoPE，所有头共享这一段 key。

动手：`exercises/model/mha_gqa_kvcache`。

> [!Q]
> **GQA 和 MQA 训练时怎么从 MHA 转过来？**
> GQA 论文的做法是 uptraining：把同组内各头的 K、V 投影矩阵做 mean-pooling 得到共享头，再用原预训练计算量的约 5% 继续训练。

## RoPE

把 $q$、$k$ 的维度两两配对，第 $i$ 对在位置 $m$ 处旋转角度 $m\theta_i$，$\theta_i = b^{-2i/d}$，$b$ 通常为 10000：

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

旋转矩阵满足 $R_m^\top R_n = R_{n-m}$，所以 $\langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle$，内积只依赖相对位置 $n-m$。RoPE 只作用在 Q、K 上，V 不旋转。

> [!WARN]
> 配对方式有两种约定：相邻配对（$2i$ 与 $2i+1$，原论文）和前后半配对（$i$ 与 $i+d/2$，HF 实现里的 `rotate_half`）。两者数学上等价（只差一个维度置换），但**权重是按哪种约定训练的，推理就必须用哪种**，混用会直接坏掉。这是自己手写模型时的高频 bug。

**长度外推**：训练长度之外的位置，低频维度的旋转角落在没见过的范围。

- **Position Interpolation**：把位置线性压缩 $m \to m \cdot L_{\text{train}}/L_{\text{target}}$，再少量微调。
- **NTK-aware**：增大 base $b$，相当于对低频维度插值、对高频维度基本不动。
- **YaRN**：按频率分段处理（NTK-by-parts），再加一个 attention 温度修正。
- 实际做法通常是长上下文继续训练 + 调大 base（Llama 3 用 500000）。

动手：`exercises/model/rope` 验证“内积只依赖相对位置”。

## 数值格式与混合精度

| 格式 | 指数位 | 尾数位 | 最大值 | 用途 |
|---|---|---|---|---|
| FP32 | 8 | 23 | 约 3.4e38 | 主权重、优化器状态、累加 |
| TF32 | 8 | 10 | 同 FP32 | Ampere+ 上 FP32 矩阵乘的 Tensor Core 内部格式 |
| FP16 | 5 | 10 | 65504 | 范围小，需要 loss scaling |
| BF16 | 8 | 7 | 约 3.4e38 | 与 FP32 同范围、精度低；当前训练主流 |
| FP8 E4M3 | 4 | 3 | 448 | 前向的权重和激活 |
| FP8 E5M2 | 5 | 2 | 57344 | 反向的梯度（要范围） |

- **FP16 的 loss scaling**：小梯度在 FP16 里会下溢成 0。把 loss 乘以 $S$ 再反向，更新前除回去。动态 loss scaling：出现 inf/nan 就跳过这步并把 $S$ 减半，连续若干步正常就把 $S$ 加倍。
- **BF16** 的范围和 FP32 一样，基本不需要 loss scaling，代价是精度只有约 3 位有效十进制数字。所以累加（矩阵乘的内积、梯度累积、softmax 的求和）要在 FP32 里做。
- **FP8 训练**：需要缩放因子把张量映射到可表示范围。Transformer Engine 用 per-tensor 缩放 + 历史 amax（delayed scaling）；DeepSeek-V3 用细粒度缩放（激活按 1×128 的 tile、权重按 128×128 的 block），并定期把 Tensor Core 的部分和提升到 FP32 累加，以控制误差。主权重和优化器状态仍是高精度。

> [!Q]
> **RL 里为什么数值精度格外敏感？**
> 推理引擎（vLLM/SGLang）和训练框架（FSDP/Megatron）用的 kernel、并行方式、累加顺序都不同，同一个 token 的 log-prob 在两边算出来会有差异，BF16 下更明显。这会让本来 on-policy 的数据变得“轻微 off-policy”，需要 TIS 等重要性采样修正（见 RL 理论的“实践问题”一篇）。也有 2025 年的工作主张 RL 训练改用 FP16 来缩小这一差异。

## Kernel fusion

逐元素算子（RMSNorm、SwiGLU、bias + dropout + residual、RoPE、softmax、Adam 更新）几乎全是访存受限的。每多一个独立 kernel，就多一次“从 HBM 读出、写回 HBM”。把多个算子融合成一个 kernel，中间结果留在寄存器里，只读写一次：

- 例：RMSNorm 不融合要依次算平方、求均值、开方、相除、乘权重，多次读写同一张量；融合后读一次 $x$、写一次输出。
- 工具：Triton 手写、`torch.compile`（Inductor 自动生成 Triton kernel）、Liger Kernel 等现成融合库；优化器用 `foreach` / fused Adam 一次更新多个张量。
- **融合 ≠ CUDA Graphs**：融合减少 HBM 往返；CUDA Graphs 减少 CPU 端的 kernel launch 开销。两者经常一起用。

> [!Q]
> **为什么 cross-entropy 也值得融合？**
> LLM 的 logits 是 `tokens × vocab`，词表十几万，物化 FP32 logits 动辄几十 GB。融合的 cross-entropy kernel 分块计算 logsumexp 和目标 token 的 logit，不物化完整的 softmax，显存和时间都省。RL 里算 log-prob 和 entropy 也是同样的道理。
