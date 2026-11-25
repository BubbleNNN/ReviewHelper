---
title: 显存账：参数、梯度、优化器、激活与 KV cache
summary: 把训练和推理的显存拆成几项逐项估算，能当场回答“这个模型在这些卡上放不放得下”。
tags: 显存, ZeRO, KV cache, 估算
order: 20
---

## 训练显存的四大块

<div class="cols"><div><h4>模型状态</h4><p>参数、梯度、优化器状态。和参数量 Ψ 成正比，和 batch 无关。</p></div><div><h4>激活</h4><p>前向存下来给反向用的中间结果。和 batch × 序列长度 × 层数成正比。</p></div><div><h4>临时 buffer</h4><p>通信 bucket、all-gather 出来的完整参数、logits、kernel workspace。</p></div><div><h4>碎片</h4><p>caching allocator 里空闲但无法拼成大块的显存。</p></div></div>

## 混合精度 Adam：每参数 16 字节

标准的混合精度训练（bf16/fp16 计算 + fp32 主权重）每个参数占：

| 项 | 精度 | 字节 |
|---|---|---|
| 参数（前向/反向用） | bf16 | 2 |
| 梯度 | bf16 | 2 |
| fp32 主权重（master weights） | fp32 | 4 |
| Adam 一阶动量 $m$ | fp32 | 4 |
| Adam 二阶动量 $v$ | fp32 | 4 |
| **合计** | | **16** |

ZeRO 论文把后三项记为 $K\Psi$，$K=12$。所以 7B 模型光模型状态就是 $16 \times 7 = 112$ GB，一张 80 GB 的卡放不下全量微调。

> [!WARN]
> 很多实现会把梯度累加在 fp32 里（Megatron 的 `main_grad`），这时每参数是 18 字节甚至 20 字节。面试时先说 16 字节这个“教科书数”，再补一句实际框架可能更多，显得你真的看过实现。

## ZeRO 三个阶段

设数据并行度为 $N_d$，ZeRO 依次切分优化器状态、梯度、参数：

| 阶段 | 切分内容 | 每卡模型状态显存 | 通信量（相对 DDP） |
|---|---|---|---|
| DDP | 不切 | $(2+2+K)\Psi = 16\Psi$ | 1×（all-reduce） |
| ZeRO-1 | 优化器状态 | $4\Psi + K\Psi/N_d$ | 1× |
| ZeRO-2 | + 梯度 | $2\Psi + (2+K)\Psi/N_d$ | 1×（reduce-scatter 替代 all-reduce） |
| ZeRO-3 | + 参数 | $(2+2+K)\Psi/N_d$ | 1.5×（前后向各多一次 all-gather） |

ZeRO 论文里的经典数字（$\Psi=7.5$B，$N_d=64$）：DDP 120 GB → ZeRO-1 31.4 GB → ZeRO-2 16.6 GB → ZeRO-3 1.9 GB。

> [!Q]
> **为什么 ZeRO-1/2 的通信量和 DDP 一样，ZeRO-3 多 50%？**
> DDP 的 all-reduce 等价于 reduce-scatter + all-gather，总通信量约 $2\Psi$。ZeRO-2 只做 reduce-scatter 拿到自己那份梯度，更新完再 all-gather 参数，还是 $2\Psi$。ZeRO-3 的参数平时是切开的，前向要 all-gather 一次，反向又要 all-gather 一次（前向用完就丢了），再加梯度的 reduce-scatter，共 $3\Psi$。

### 完整算账：7B 和 70B

**7B 全量微调，8 卡，混合精度 Adam：**

- ZeRO-1：$4 \times 7 + 12 \times 7/8 = 28 + 10.5 = 38.5$ GB
- ZeRO-2：$2 \times 7 + 14 \times 7/8 = 14 + 12.25 = 26.25$ GB
- ZeRO-3：$16 \times 7/8 = 14$ GB

剩下的显存给激活、通信 buffer 和 logits。

**70B，64 卡，ZeRO-3/FSDP：** 模型状态 $16 \times 70 / 64 = 17.5$ GB/卡。但前向时要把一层的完整参数 all-gather 出来：70B 有 80 层，每层约 0.85B 参数，bf16 约 1.7 GB，开 prefetch 时同时存在 2 层，约 3.4 GB。再加激活，80 GB 的卡一般够用。同样的模型用 ZeRO-1 每卡要 $4 \times 70 = 280$ GB，完全不可能。

## 激活显存

Korthikanti et al.（Megatron，2022）给出 GPT 结构每层激活（fp16，无并行）约为：

$$\text{Act}_{\text{layer}} \approx s\,b\,h\left(34 + \frac{5\,a\,s}{h}\right)\ \text{bytes}$$

其中 $s$ 为序列长度，$b$ 为 micro-batch，$h$ 为 hidden size，$a$ 为头数。第二项来自 attention score / softmax / dropout mask，是 $s^2$ 级别的。FlashAttention 不落地这些矩阵，所以这一项基本消失。SwiGLU、无 dropout 的模型系数会略有不同，但量级一致。

**例**：7B（$h=4096$，32 层），$s=4096$，$b=1$，用 FlashAttention：每层约 $34 \times 4096 \times 4096 \approx 0.57$ GB，32 层约 18 GB。

### Activation checkpointing

- **Full recompute**：每层只存输入（$2sbh$ 字节），反向时重跑这层前向。显存降到约 $2sbhL$，代价是多一次前向，计算从 $6N$ 变成 $8N$，约多 33%。
- **Selective recompute**：只重算 attention 里“占显存大、算得快”的部分（上式的 $5as/h$ 项），用很少的额外计算换掉大部分显存。
- **Activation offload**：把激活搬到 CPU，受 PCIe 带宽限制（Gen5 x16 单向约 64 GB/s），需要和计算重叠。

> [!TIP]
> TP + Sequence Parallel 会把激活再除以 $t$：Megatron 的 SP 把 LayerNorm、Dropout 的激活也沿序列维切开，所以每层激活变成约 $34sbh/t$。

## RL 场景特有的大头：logits

计算 log-prob 时要先得到 `(tokens, vocab)` 的 logits。词表约 15 万（如 Qwen2.5 的 152064），一个 micro-batch 有 32K 个 token 时，fp32 logits 就有 $32768 \times 152064 \times 4 \approx 20$ GB。对策：

- 按 micro-batch 算 log-prob，用 `log_softmax` + `gather` 之后立刻丢掉 logits；
- 用分块 / 融合的 cross-entropy kernel（如 Liger Kernel、verl 的 `use_fused_kernels`），不把完整 logits 物化；
- 动态 batch（verl `use_dynamic_bsz` + `ppo_max_token_len_per_gpu`）按 token 数而不是样本数切 micro-batch。

## KV cache

推理（包括 RL 的 rollout）时显存大头是 KV cache：

$$\text{KV bytes} = 2 \times L \times H_{kv} \times d_{\text{head}} \times \text{bytes} \times \text{tokens}$$

“2”是 K 和 V，`tokens` 是所有序列的上下文长度之和。

| 模型 | 层数 | KV 头数 | head dim | 每 token KV（bf16） |
|---|---|---|---|---|
| Llama-2-7B（MHA） | 32 | 32 | 128 | 512 KB |
| Llama-2-70B（GQA, 8 组） | 80 | 8 | 128 | 320 KB |
| 假设 70B 用 MHA（64 头） | 80 | 64 | 128 | 2.5 MB |

GQA 让 70B 的 KV 缩小了 8 倍。**MLA**（DeepSeek-V2/V3）更进一步：每层只缓存一个压缩后的 latent $c_{KV}$（维度 $d_c=512$）加一个解耦的 RoPE key（维度 64），每层每 token 只有 576 个元素，而同配置 MHA（128 头 × 128 维）是 $2 \times 128 \times 128 = 32768$ 个。

**例**：70B 服务 32 条并发、每条 4K 上下文：$32 \times 4096 \times 320\ \text{KB} \approx 40$ GB。加上 140 GB 的 bf16 权重，至少需要 TP=4 的 H100（每卡 35 GB 权重 + 10 GB KV + 余量）。

> [!Q]
> **vLLM 的 `gpu_memory_utilization=0.5` 是什么意思？**
> vLLM 先 profile 一次前向的峰值显存，然后把“总显存 × 0.5 − 权重 − 峰值激活”全部预分配给 KV cache 的 block 池。在 RL 里推理和训练共用一张卡（colocate），所以不能像纯推理那样设成 0.9。SGLang 的对应参数是 `mem_fraction_static`，语义上是静态分配（权重 + KV 池）占总显存的比例。

## 一个 RL colocate 的显存预算

把上面几项合起来，看一个典型场景：7B 模型做 GRPO，单节点 8 张 80 GB 卡，训练用 FSDP（ZeRO-3），rollout 用 vLLM，两者同卡（colocate）。

| 阶段 | 显存里有什么 | 每卡估算 |
|---|---|---|
| 训练 | 模型状态 $16 \times 7 / 8$ | 14 GB |
| | 一层完整参数（all-gather）+ 激活（checkpointing）+ logits 分块 | 十几 GB，视 micro-batch 的 token 数而定 |
| rollout | vLLM 加载的完整 bf16 权重（rollout TP=1 时） | 14 GB |
| | KV cache 池：`gpu_memory_utilization=0.5` 即总共最多用 40 GB | 约 40 − 14 − 峰值激活 ≈ 20+ GB |

能放下的前提是两个阶段**不同时占满**：rollout 时训练侧的参数和优化器状态 offload 到 CPU（或至少释放梯度和激活），训练时 vLLM 释放 KV cache、sleep 掉权重。20 GB 的 KV 池按每 token 0.5 MB（Llama-2-7B 式 MHA）算只能放约 4 万个 token，一个 batch 里 256 条 × 8K 的回复（约 200 万 token）要分很多轮才能跑完。所以 rollout 的 KV 容量往往直接决定了 RL 的吞吐，换成 GQA 模型（KV 小 4 到 8 倍）会好得多。

## 碎片与 PyTorch caching allocator

`cudaMalloc/cudaFree` 很慢且会同步，PyTorch 用 caching allocator：释放的块不还给驱动，留着下次复用。

- `torch.cuda.memory_allocated()`：张量实际占用；`memory_reserved()`：allocator 从驱动拿到的总量（含缓存）。
- **碎片**：reserved 里有很多空闲小块，但拼不出一个连续大块，于是 OOM，报错里会出现 “reserved but unallocated” 很大。变长序列（RL rollout 长度差异大）特别容易碎。
- 对策：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（段可扩展，大幅减轻碎片）；固定 shape / 按 token 数分桶；在训练和推理切换时 `torch.cuda.empty_cache()` 把缓存还给驱动，让 vLLM/SGLang 能拿到。

> [!WARN]
> colocate 模式下，推理引擎是另一个“显存使用者”，它看不到 PyTorch allocator 的缓存。训练完不 `empty_cache`、推理完不释放 KV（verl 的 `free_cache_engine` / sleep），两边就会互相挤爆。

> [!Q]
> **估算题：13B 模型 LoRA 微调（rank 16，只训 attention），单卡 80 GB 够吗？**
> 冻结的 bf16 权重 26 GB；LoRA 参数只有几千万，优化器状态可忽略；激活看序列长度，用 gradient checkpointing 后几 GB。够用。关键是冻结权重不需要梯度和优化器状态，16 字节/参数只作用在 LoRA 参数上。

动手：`exercises/system/memory_estimator` 实现 ZeRO 各阶段与 KV cache 的显存计算。
