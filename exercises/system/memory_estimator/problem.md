---
title: 显存估算器：ZeRO、KV Cache 与 MFU
difficulty: 易
tags: 显存, ZeRO, KV cache, 估算
order: 14
source: ZeRO (Rajbhandari et al., SC'20) Figure 1 · vLLM 显存规划
---

面试里的估算题几乎都能拆成下面几个公式。实现 4 个函数，单位一律是**字节**、**FLOPs**，不做 GiB 换算。

**1. 训练时每卡的模型状态显存**（混合精度 Adam，参数量 $\Psi$，数据并行度 $N_d$，$K$ 默认 12）

- fp16/bf16 参数 $2\Psi$，fp16/bf16 梯度 $2\Psi$
- 优化器状态 $K\Psi$：fp32 master 参数、一阶动量、二阶动量各 $4\Psi$

| stage | 切分对象 | 每卡显存 |
|---|---|---|
| 0 | 不切 | $(2+2+K)\Psi$ |
| 1 | 优化器状态 | $2\Psi + 2\Psi + K\Psi/N_d$ |
| 2 | 优化器 + 梯度 | $2\Psi + (2+K)\Psi/N_d$ |
| 3 | 优化器 + 梯度 + 参数 | $(2+2+K)\Psi/N_d$ |

```python
zero_model_states_bytes(num_params, dp, stage, k=12) -> float
```

**2. KV cache 大小**（K 和 V 各一份）

$$\text{bytes} = 2 \times L \times H_{kv} \times d_{head} \times S \times B \times e$$

其中 $S$ 是序列长度，$B$ 是 batch，$e$ 是每个元素的字节数（bf16 为 2）。

```python
kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, batch=1, dtype_bytes=2) -> int
```

**3. 能同时服务多少条序列**：给定可用于 KV 的显存 `free_bytes`，每条序列长 `seq_len`。如果给了 `block_size`（PagedAttention），每条序列占用 $\lceil S/\text{block size}\rceil$ 个块，每块存 `block_size` 个 token 的 KV。

```python
max_concurrent_seqs(free_bytes, num_layers, num_kv_heads, head_dim, seq_len, dtype_bytes=2, block_size=None) -> int
```

**4. MFU**：训练 FLOPs 约为 $6ND$（前向 $2N$、反向 $4N$ 每 token）。

```python
mfu(num_params, tokens_per_sec, num_gpus, peak_flops_per_gpu) -> float  # = 6·N·tps / (gpus·peak)
```

<!-- solution -->

### 讲解

**ZeRO 论文的经典数字**（$\Psi=7.5$B，$N_d=64$）：stage 0 为 120 GB，stage 1 为 31.4 GB，stage 2 为 16.6 GB，stage 3 为 1.9 GB。可以看出 stage 1 一下省掉大头，因为优化器状态占 $12/16$。

**KV cache 的直觉。** Llama-2-7B（32 层、32 个 KV head、head_dim 128、bf16）每个 token 占 $2\cdot 32\cdot 32\cdot 128\cdot 2 = 512$ KiB，4K 上下文一条序列就是 2 GiB。Llama-3-70B 用 GQA（8 个 KV head，80 层），每 token 只要 320 KiB。这就是 GQA / MLA 的主要动机：decode 的瓶颈在读 KV。

**块粒度的浪费。** PagedAttention 只在每条序列的最后一个块里有内部碎片，最多浪费 `block_size - 1` 个 token 的空间，远好于按最大长度预留。

**MFU。** 例如 7B 模型、64 卡 H100（bf16 dense 峰值约 989 TFLOPS）、整个集群每秒 0.6M tokens：$6\cdot 7\times10^9\cdot 6\times10^5/(64\cdot 989\times10^{12}) \approx 0.40$。大模型预训练做到 40%–55% 已经算好；算出来超过 1 说明数据有错。注意 activation checkpointing 的重算不计入 MFU（计入的话叫 HFU）。

**易错点。**

- 激活显存不在 ZeRO 公式里。长序列时激活常常比模型状态还大，要靠 checkpointing / SP / CP 解决。
- KV cache 别忘了乘 2（K 和 V）。GQA 用的是 KV head 数，不是 query head 数。
- GB 与 GiB：论文用的是 $10^9$。

> [!Q]
> 70B 模型，bf16 + Adam，ZeRO-3，64 卡，每卡模型状态多少？还能放多大的激活？
>
> $16 \times 70\text{B}/64 = 17.5$ GB。80 GB 卡还剩约 60 GB 给激活、通信 buffer 和碎片。但 ZeRO-3 每层计算前要临时 all-gather 完整参数，峰值还要加上最大一层（或一个 FSDP unit）的完整参数。

> [!Q]
> 推理时 batch 能开多大？
>
> (显存 − 权重 − 激活/workspace) / 每序列 KV。vLLM 里 `gpu_memory_utilization` 控制的就是这块预算。所以 GQA、KV 量化（FP8 KV）、更短的上下文都能直接换来更大的 batch。
