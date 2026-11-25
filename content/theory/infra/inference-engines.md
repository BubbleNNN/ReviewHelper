---
title: 推理引擎：PagedAttention、Continuous Batching 与 PD 分离
summary: vLLM / SGLang 这类引擎如何管理 KV cache、调度请求、复用前缀，以及衡量它们的指标。
tags: vLLM, SGLang, KV cache, 调度
order: 60
---

## 先定指标

| 指标 | 含义 | 主要受什么影响 |
|---|---|---|
| TTFT | Time To First Token，请求到达到第一个 token | 排队 + prefill |
| TPOT / ITL | 每个输出 token 的间隔 | decode 步长，受 batch 大小影响 |
| E2E 延迟 | $\text{TTFT} + \text{TPOT}\times(n-1)$ | 两者都有 |
| 吞吐 | tokens/s 或 requests/s | batch 越大越高 |
| Goodput | 满足 SLO（如 TTFT < 2s 且 TPOT < 50ms）的请求速率 | 吞吐与延迟的平衡 |

核心矛盾：**batch 越大，吞吐越高，但每个请求的 TPOT 越差**。引擎的大部分设计都是在一定 SLO 下尽量把 batch 做大，而 batch 能做多大主要取决于 KV cache 能装下多少 token。

## KV cache 与 PagedAttention

自回归生成每步都要用到所有历史 token 的 K、V，所以把它们缓存起来，每步只为新 token 计算 K、V。KV 大小公式见“显存账”。

早期系统按“最大长度”给每个请求预留一段连续显存，问题是：

- **内部碎片**：预留 4096 个位置，实际只生成 300 个；
- **外部碎片**：不同长度的空洞拼不起来；
- **无法共享**：同一 prompt 的多个采样各存一份。

vLLM 论文估计已有系统的 KV 显存浪费在 60%–80%。**PagedAttention** 借鉴操作系统的分页：

- KV cache 切成固定大小的 **block**（默认 16 个 token），物理上不连续；
- 每个序列有一张 **block table**，记录逻辑块到物理块的映射，attention kernel 按表查找；
- 只有最后一个块可能没填满，浪费不到一个块（论文报告 4% 以内）；
- **共享与 copy-on-write**：并行采样 / beam search 时多个序列共享前缀块，用引用计数管理；某个序列要往共享块里写时才复制一份。
- **抢占**：块不够时把一些请求踢出去，要么把 KV 换出到 CPU（swap），要么丢掉、之后重算（recompute）。

<div class="flow"><span>逻辑块 0,1,2</span><i>→</i><span class="f-infra">block table<small>[7, 2, 9]</small></span><i>→</i><span>物理块 7, 2, 9<small>任意位置</small></span></div>

动手：`exercises/system/paged_kv` 实现带引用计数和 copy-on-write 的块分配器。

## Continuous Batching

**静态 batching**：凑齐一个 batch 一起跑，要等最长的那条生成完，整个 batch 才能换人，短请求的位置空等。

**Continuous batching**（Orca 提出的 iteration-level scheduling）：**每个 decode 步之后**都重新调度，生成完的请求立刻退出，新请求立刻加入。配合 PagedAttention，batch 大小可以随时变化，GPU 一直是满的。

调度器每一步要决定：

1. 哪些 running 请求继续 decode（每个占 1 个 token 预算）；
2. 有没有空余的 block 和 token 预算（`max_num_batched_tokens`、`max_num_seqs`）接纳 waiting 队列里的新请求做 prefill；
3. block 不够时抢占谁（通常抢占最晚到达的）。

动手：`exercises/system/continuous_batching` 比较静态与 continuous batching 的完成时间。

### Chunked prefill

一个 32K token 的长 prompt 进来，整段 prefill 要算好几百毫秒，这期间所有正在 decode 的请求都被卡住，TPOT 出现尖刺。**Chunked prefill**（Sarathi-Serve）把长 prefill 切成块，每一步在 token 预算内混合“一块 prefill + 所有 decode token”一起算：

- decode 不再被长 prefill 阻塞，TPOT 尾延迟显著改善；
- prefill 是算力受限的、decode 是访存受限的，混在一起正好互补；
- 代价是长 prompt 的 TTFT 略微变长。

## 前缀复用：Prefix Caching 与 RadixAttention

很多请求有共同前缀：system prompt、few-shot 示例、多轮对话历史、**RL 里 GRPO 对同一个 prompt 采样 n 次**。

- **vLLM Automatic Prefix Caching**：对每个**满**的 block 计算哈希（父块哈希 + 本块 token），相同哈希直接复用已有的 KV 块；引用计数为 0 的块进入可淘汰队列，按 LRU 回收。
- **SGLang RadixAttention**：用一棵 radix tree 按 token 序列组织所有缓存的 KV，新请求做最长前缀匹配；淘汰时按 LRU 删叶子节点。调度上做 **cache-aware scheduling**：优先调度匹配前缀最长的请求，提高命中率。

动手：`exercises/system/radix_cache`；也可以回看 `exercises/algo/lru_cache`。

> [!Q]
> **RL 的 rollout 为什么特别吃 prefix caching？**
> GRPO 每个 prompt 采 $n$（如 8 或 16）条回复，prompt 部分完全相同，只需 prefill 一次；多轮 agent 每一轮都带着前面的全部历史，复用历史的 KV 能省掉大量重复 prefill。注意每次权重更新后旧的 KV 都失效了，缓存要清空。

## PD 分离（Prefill/Decode Disaggregation）

prefill 算力受限、decode 访存受限，放在同一批卡上会互相干扰（chunked prefill 只能缓解），而且两者适合的并行配置不同。**PD 分离**把它们放到不同的 GPU 池：

<div class="flow"><span class="f-infra">Prefill 实例<small>算 prompt，产出 KV</small></span><i>→</i><span>KV 传输<small>RDMA / NVLink</small></span><i>→</i><span class="f-accent">Decode 实例<small>大 batch 持续生成</small></span></div>

- **好处**：消除干扰；TTFT 和 TPOT 可以分别优化；两个池可以独立扩缩容、用不同的 TP/EP 配置。
- **代价**：KV 传输占带宽（长 prompt 的 KV 很大）；系统更复杂，要做 KV 的路由和负载均衡。
- **代表系统**：DistServe、Splitwise、Mooncake（Kimi，以 KV cache 为中心，用 CPU 内存/SSD 组成分布式 KV 池）。DeepSeek-V3 的部署也是 PD 分离，技术报告里 prefill 最小部署单元是 32 卡、decode 是 320 卡（大规模 EP；数字待核实）。

## CUDA Graphs

decode 每一步有几百个小 kernel，每个 launch 要几微秒，小 batch 时 CPU 发射 kernel 的时间可能比 GPU 计算还长。**CUDA Graphs** 把一整步的 kernel 序列录制下来，之后一次回放：

- 图里的 shape 和显存地址是固定的，所以引擎会按若干个 batch 大小（如 1, 2, 4, 8, …, 256）分别录制，运行时把 batch pad 到最近的录制大小；
- prefill 的 shape 变化大，通常不录，或者只把 attention 以外的部分分段录制（vLLM 的 piecewise CUDA graph）；
- 录制本身要时间和显存。verl 的 rollout 配置里 `enforce_eager` 默认是 True，即默认关闭 CUDA Graphs。

## 量化

| 方案 | 做法 | 主要收益 |
|---|---|---|
| W4A16（GPTQ / AWQ） | 权重 4 bit，计算时反量化成 16 bit | decode 访存受限，读的权重少了 4 倍 |
| W8A8 INT8（SmoothQuant） | 用逐通道缩放把激活的离群值“搬”到权重上，再都量化成 INT8 | INT8 GEMM 更快，利好 prefill |
| FP8 W8A8 | Hopper 上 E4M3 权重 + 激活 | 精度损失很小，算力翻倍 |
| KV cache FP8/INT8 | 量化 K、V | KV 显存减半，可并发更多请求 |

## vLLM 与 SGLang

两者是 verl 支持的两个 rollout 引擎，核心技术高度重合（PagedAttention 式的分页 KV、continuous batching、chunked prefill、prefix caching、CUDA Graphs），侧重点不同：

- **vLLM**：PagedAttention 的发源地，模型和硬件覆盖面广。V1 引擎重构了调度器，prefill 和 decode 统一按 token 预算调度，默认开启 chunked prefill 和 prefix caching。
- **SGLang**：RadixAttention 前缀树缓存、cache-aware 调度、结构化输出（约束解码）、把 CPU 端调度和 GPU 计算重叠的 overlap scheduler，对 DeepSeek 系列（MLA、大 EP）的优化做得较早。

> [!Q]
> **线上 TPOT 突然变差，可能是什么原因？**
> ①并发上来了，batch 变大，每步变慢；②有超长 prompt 的 prefill 插进来，没开或没调好 chunked prefill；③KV cache 不够，频繁抢占和重算；④prefix cache 命中率下降导致 prefill 变多；⑤开了投机解码，但高负载下验证开销超过了收益。

> [!Q]
> **给定一张 80 GB 卡跑 8B 模型（bf16），大概能并发多少条 4K 上下文的请求？**
> 权重约 16 GB。按 Llama-3-8B（32 层、8 个 KV 头、head dim 128）算，每 token KV $= 2\times32\times8\times128\times2$ B $=128$ KB，4K 上下文一条约 0.5 GB。留出约 10 GB 给激活和余量，剩约 54 GB 给 KV，大约 100 条。这类估算题要先写出公式，再代数。

动手：`exercises/model/sampling` 实现 temperature / top-k / top-p 采样。
