---
title: 性能开关：去 padding、动态 batch、序列并行、显存卸载
summary: verl 训练侧和推理侧最常调的性能配置，每个开关解决什么问题、代价是什么。
tags: verl, 性能, 显存, FSDP
order: 40
---

RL 训练的数据有两个特点：**response 长度方差极大**（几十到几万 token），以及**同一批 GPU 要轮流装下训练状态和推理引擎**。verl 的性能开关基本都围绕这两点。

## 训练侧：别为 padding 付钱

### Remove padding / sequence packing

`actor_rollout_ref.model.use_remove_padding=True`：把一个 micro-batch 里所有序列的有效 token 拼成一条长序列（形状从 `(B, S)` 变成 `(1, ΣL_i)`），用 FlashAttention 的 varlen 接口（`cu_seqlens` 标记边界）保证不同序列之间不互相 attend。

- 收益：padding 比例越高收益越大，长尾分布下常见 2× 以上的训练吞吐。
- 前提：模型的 attention 走 flash-attn varlen 路径，位置编码按每条序列重新从 0 开始（`position_ids` 也要打包）。

### Dynamic batch size

`actor.use_dynamic_bsz=True` + `actor.ppo_max_token_len_per_gpu`（默认 16384）：不再固定每个 micro-batch 的样本数，而是按 **token 预算**把样本装箱（`rearrange_micro_batches`）。短序列一次能装很多，长序列一次只装一两条，每个 micro-batch 的显存和计算量接近。

- log-prob 计算有独立的预算 `log_prob_max_token_len_per_gpu`（默认继承 actor 的值）。前向不存激活，可以设得更大。
- critic 的 `ppo_max_token_len_per_gpu` 默认 32768。
- 经验值：`ppo_max_token_len_per_gpu` ≥ `n × (max_prompt + max_response)` 量级，至少要装得下一条最长样本。

### DP 间负载均衡

即使每卡 token 数接近，attention 的 $O(s^2)$ 仍会让“少数长序列”的那张卡更慢。V1 trainer 在 driver 上用 `calculate_workload`（$24576\,s+s^2$，按 7B、hidden=4096 标定）估算每条序列的计算量，再用 Karmarkar–Karp 差分法（`get_seqlen_balanced_partitions`）分成工作量接近的 DP 份。动手：`exercises/system/seqlen_balance`。

<div class="cols"><div><h4>remove padding</h4><p>消掉一个 micro-batch 内部的 padding。</p></div><div><h4>dynamic bsz</h4><p>让每个 micro-batch 的 token 数接近。</p></div><div><h4>balance batch</h4><p>让每个 DP rank 的总计算量接近，减少同步等待。</p></div></div>

## 训练侧：长序列装不下

| 开关 | 默认 | 作用 | 代价 |
|---|---|---|---|
| `model.enable_gradient_checkpointing` | true | 每层只存输入，层内激活在反向时重算；按 Megatron 的估算，每层激活从约 $sbh(34+5as/h)$ 字节降到约 $2sbh$ | 多一次前向，约 +30% 计算 |
| `model.enable_activation_offload` | false | 激活卸载到 CPU | PCIe 带宽成为瓶颈 |
| `actor.ulysses_sequence_parallel_size` | 1 | DeepSpeed-Ulysses 序列并行：all-to-all 把序列维切分换成 head 维切分，attention 时每卡算完整序列的部分 head | 需要 head 数能被 SP 整除；多两次 all-to-all |
| `fsdp_config.param_offload` / `optimizer_offload` | false | 参数 / 优化器状态卸载到 CPU | 每步要来回搬，慢；ref 模型常开 `param_offload` |
| `model.use_fused_kernels` | false | 融合 lm_head + log-prob/entropy 计算，避免物化 `(T, V)` 的 logits | 依赖特定实现后端 |
| `model.lora_rank` | 0 | >0 时启用 LoRA，只训低秩适配器 | 表达能力受限 |

> [!Q]
> **为什么算 log-prob 时 logits 特别吃显存？怎么省？**
> logits 形状是 `(tokens, vocab)`，vocab 动辄 15 万。32k token × 152k × 4 字节（fp32 softmax）≈ 19 GB，单这一项就可能 OOM。办法：按 token 分块算 log-softmax；或者融合 kernel，让 lm_head 的输出边算边规约，只留下每个 token 的 log-prob 和 entropy，不物化完整 logits。

## 推理侧：rollout 的吞吐

| 开关 | 默认 | 说明 |
|---|---|---|
| `rollout.name` | 必填 | `vllm` / `sglang` / `trtllm` / `hf` |
| `rollout.gpu_memory_utilization` | 0.5 | vLLM 里是**占总显存**的比例（权重 + KV cache）；SGLang 对应 `mem_fraction_static` |
| `rollout.tensor_model_parallel_size` | 2 | 推理 TP；模型越大越需要，TP 太大会增加通信、降低单卡效率 |
| `rollout.max_num_batched_tokens` / `max_num_seqs` | 8192 / 256 | 调度器每步的 token 预算和最大并发序列 |
| `rollout.enable_chunked_prefill` / `enable_prefix_caching` | true / true | 长 prompt 分块 prefill；GRPO 同一 prompt 的 n 条采样共享前缀 KV |
| `rollout.enforce_eager` | false | 关掉 CUDA Graph（调试用）；开着 CUDA Graph 能显著降低 decode 的 kernel launch 开销 |
| `rollout.free_cache_engine` | true | 生成结束释放 KV cache，把显存还给训练 |
| `rollout.calculate_log_probs` | true | 让推理引擎返回采样 token 的 log-prob，rollout correction 需要 |

> [!TIP]
> GRPO 的 `rollout.n` 条采样共用一个 prompt，开 prefix caching 后 prompt 只 prefill 一次。prompt 很长的任务（长文档、agent 多轮）收益最明显。

## 时分复用时的显存账

共置模式下同一张卡的显存会经历两个阶段：

<div class="flow"><span class="f-infra">rollout 阶段<small>推理权重 + KV cache（gpu_memory_utilization）</small></span><i>⇄</i><span class="f-rl">训练阶段<small>FSDP 分片的参数 + 梯度 + 优化器 + 激活</small></span></div>

切换时：rollout 结束 `sleep`（释放 KV cache、丢弃或卸载推理权重）→ 训练 → `update_weights` 把新权重发给推理引擎并 `wake_up`。如果训练状态太大，就开 param/optimizer offload，用时间换显存。

> [!Q]
> **OOM 了，按什么顺序排查？**
> 先看是 rollout 阶段还是训练阶段（日志里的 timing 名字）。rollout OOM：降 `gpu_memory_utilization`、`max_num_seqs`，加推理 TP。训练 OOM：降 `ppo_max_token_len_per_gpu` / micro batch，确认 gradient checkpointing 开着，开 remove padding，长序列上 Ulysses SP，最后才考虑 offload。还要确认 rollout 结束后 KV cache 真的释放了（`free_cache_engine`）。

> [!Q]
> **remove padding 之后为什么还要 dynamic bsz？**
> remove padding 只去掉 padding，但固定 micro batch size 时，碰上一批全是长序列的样本 token 总数仍会暴涨，显存峰值不可控；碰上全是短序列又吃不满 GPU。dynamic bsz 按 token 预算装箱，两头都照顾到。

> [!SRC]
> 装箱：`verl/utils/seqlen_balancing.py`（`rearrange_micro_batches`、`karmarkar_karp`、`calculate_workload`）；默认值：`verl/trainer/config/_generated_ppo_trainer.yaml`。
