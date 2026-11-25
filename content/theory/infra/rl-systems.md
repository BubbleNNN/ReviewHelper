---
title: RL 训练系统：Colocate、权重同步与异步
summary: 从系统角度看 LLM 的 RL 训练：时间花在哪、训练和推理怎么摆放、权重怎么同步、长尾和 staleness 怎么处理。
tags: RL 系统, verl, 异步 RL, 权重同步
order: 80
---

## 一个 RL step 由什么组成

<div class="flow"><span class="f-infra">rollout<small>推理引擎自回归生成</small></span><i>→</i><span>reward<small>规则 / 沙箱 / RM</small></span><i>→</i><span>old / ref log-prob<small>训练框架前向</small></span><i>→</i><span>values<small>critic 前向</small></span><i>→</i><span class="f-rl">update<small>critic + actor 反向</small></span><i>→</i><span class="f-infra">weight sync<small>新权重给推理引擎</small></span></div>

和预训练相比，RL 系统的特殊之处在于**一个 step 里同时有推理负载和训练负载**，两者对硬件的需求完全不同：

- **rollout** 是 decode 为主、访存受限的推理任务，回复长度差异极大，长 CoT 任务一条可能几万 token。经验上它常常占一个 step 的一半以上时间。
- **训练**是算力受限的大 batch 前向 + 反向，需要放下优化器状态。
- 中间还穿插着 reward 计算（可能要跑代码沙箱或调用另一个模型）。

所以 RL 系统设计围绕三个问题：两种负载怎么摆放？权重怎么从训练侧同步到推理侧？怎么不让 GPU 空等？

## Colocate 还是 Disaggregate

<div class="cols"><div><h4>Colocate（时分复用）</h4><p>训练和推理用同一批 GPU，轮流使用。verl 的 hybrid engine 默认如此。</p><p>✓ 同步 on-policy 时没有闲置卡；权重同步在卡内完成，很快；资源管理简单。</p><p>✗ 两边要轮流腾显存（offload、sleep、释放 KV）；生成和训练只能串行；切换有开销。</p></div><div><h4>Disaggregate（分池）</h4><p>训练池和推理池是不同的 GPU。</p><p>✓ 两边可以各自选最合适的并行配置和硬件；配合异步可以让生成和训练同时进行。</p><p>✗ 同步模式下两边互相等待，会有大段空闲；权重要跨节点传输。</p></div></div>

> [!KEY]
> 同步、on-policy 的设置下，colocate 的 GPU 利用率更高；一旦引入异步（允许一定的 off-policy），disaggregate 能把生成和训练重叠起来，优势就显现了。verl V1 trainer 的三种模式正好对应这几种组合：`sync`（colocate，同步）、`colocate_async`（colocate，开启 partial rollout）、`separate_async`（分池，训练侧空闲时可以切过去帮忙生成）。

### Colocate 时的显存腾挪

一张卡上要轮流放下“训练状态”和“推理状态”：

1. rollout 前：训练侧把参数、梯度、优化器状态 offload 到 CPU（或至少释放梯度和激活），推理引擎 **wake up**，加载最新权重、重新分配 KV cache；
2. rollout 后：推理引擎 **sleep**，释放 KV cache（verl `free_cache_engine=True`）、把权重 offload 或丢弃；训练侧再把状态搬回来；
3. 两边用的是不同的显存分配器，要配合 `torch.cuda.empty_cache()`，否则 PyTorch 缓存着的显存推理引擎拿不到（见“显存账”）。

## 权重同步

每次 actor 更新后，推理引擎要拿到新权重才能继续 on-policy 生成。难点在于**两边的并行布局不同**：训练侧可能是 FSDP 分片或 Megatron TP/PP，推理侧是 vLLM/SGLang 的 TP。所以同步分两步：先把训练侧的分片**重排**成推理侧需要的布局，再**传输**。

- **同卡（colocate）**：在 GPU 上把参数按层 all-gather 成完整张量（或直接按推理 TP 布局切好），然后通过 CUDA IPC 把显存句柄交给推理进程（零拷贝），或者在同一进程里直接 `load_weights`。verl 论文里的 3D-HybridEngine 就是为了让这一步没有冗余显存、通信量最小。
- **跨卡（disaggregate）**：训练侧 rank 通过 NCCL broadcast / 点对点把权重发给推理 worker，按 bucket 流式发送，避免一次占用太多显存。
- **verl 的 checkpoint engine** 把这件事抽象成 `send_weights` / `receive_weights` / `get_weights` 三个接口，后端包括：

| 后端 | 通信方式 | 适用场景 |
|---|---|---|
| naive | torch.distributed all_gather | 训推同卡（colocate），on-policy |
| nccl / hccl | all_gather + broadcast | 训推分离、集群固定 |
| nixl / mooncake | all_gather + ring p2p | 训推分离，支持弹性扩缩、故障容忍、异构硬件 |
| kimi_ckpt_engine | Mooncake + NCCL/HCCL：p2p + broadcast | 训推分离 |

量级感：70B 模型 bf16 权重 140 GB，跨节点单卡约 50 GB/s，靠多卡并行传输和流水化才能把同步压到秒级到十几秒级。verl 还在做 delta weight sync（只传变化量）。

> [!Q]
> **为什么不直接把 checkpoint 存盘再让推理引擎读？**
> 能用，但慢：大模型权重上百 GB，写盘再读盘是几十秒到分钟级，而一个 RL step 可能只有几分钟。走 GPU 间的 NCCL / RDMA / CUDA IPC 能快一到两个数量级。

## 长尾 rollout

回复长度是重尾分布：大部分回复几千 token，少数几万。同步 rollout 要等最长的那条生成完，最后一段时间 batch 里只剩几条请求，GPU 几乎空转。对策：

- **Partial rollout**（Kimi k1.5 提出）：每次迭代给生成设一个 token 预算，没写完的回复先存起来，下一次迭代用新策略接着写。代价是一条回复的不同片段来自不同版本的策略（轻微 off-policy）。
- **异步**：生成和训练解耦，见下一节。
- **过采样 + 提前截止**：多发一些 prompt，凑够数量就停。要小心：这会系统性地偏向短回复，等于改变了数据分布。
- **推理侧本身的优化**：continuous batching 能让短请求结束后立即补进新请求；长尾阶段 batch 小，也是投机解码最有效的时候。

## 异步 RL 与 staleness

<div class="flow"><span class="f-infra">rollout workers<small>持续生成，用当时最新的权重</small></span><i>→</i><span>replay buffer<small>记录每条样本的策略版本</small></span><i>→</i><span class="f-rl">trainer<small>取数训练，定期推送新权重</small></span><i>↺</i></div>

- **One-step off-policy**：训练第 $t$ 批数据时，同时用第 $t$ 步之前的权重生成第 $t+1$ 批。数据只“过时”一步，实现简单，能把生成和训练重叠起来。
- **Fully async**：生成侧持续不断地生成，训练侧从 buffer 里取数。用 **staleness 上限 $\eta$** 控制：策略版本落后超过 $\eta$ 的样本丢弃或不予采样。
- **必须做 off-policy 修正**：生成时记录行为策略的 log-prob（verl 要求 `rollout.calculate_log_probs=True`），训练时用重要性采样比修正，并截断过大的比值（TIS 等）。AReaL 提出 **decoupled PPO**：重要性权重相对“行为策略”计算，clip 则相对一个较新的“近端策略”计算，避免陈旧数据让 clip 失效。
- verl V1 的 replay buffer 基于 TransferQueue 存放轨迹，key 为 `{uid}_{session_id}_{index}`，并统计被淘汰样本的 staleness 指标。

> [!Q]
> **异步 RL 一定更好吗？**
> 不一定。它换来的是吞吐，代价是数据变成 off-policy：staleness 越大，梯度估计的偏差和方差越大，训练越容易不稳定。所以要限制 staleness、做重要性采样修正，并实测“单位墙钟时间的效果”而不是单纯看吞吐。

## 负载均衡

数据并行里，各 rank 的总 token 数不同会导致快的等慢的。

- **跨 DP rank 均衡**：verl 的 `balance_batch` 按序列长度重排样本，用 Karmarkar–Karp 划分算法（`get_seqlen_balanced_partitions`）让每个 rank 的 token 总数尽量接近。
- **动态 micro-batch**：`use_dynamic_bsz=True` 时按 token 预算 `ppo_max_token_len_per_gpu` 把序列装进 micro-batch，而不是固定样本数，长短混合时显存更稳、padding 更少。
- **去 padding**：`use_remove_padding` 把一个 micro-batch 的序列首尾拼接（packing），用 varlen 的 FlashAttention 计算，不在 padding 上浪费算力。
- **推理侧路由**：多副本之间按负载分配请求；多轮 agent 的同一个会话粘在同一个副本上（sticky session），以复用前缀 KV cache。

动手：`exercises/system/seqlen_balance` 实现按长度均衡划分和按 token 预算切 micro-batch。

## 主流 RL 框架的一句话定位

| 框架 | 一句话 |
|---|---|
| verl（字节） | HybridFlow：单 controller + Ray worker group，训推同卡的 hybrid engine；FSDP/Megatron 训练 + vLLM/SGLang 推理；见“框架 / verl”。 |
| OpenRLHF | Ray + vLLM + DeepSpeed，可以把 actor、critic、RM、ref 分别放在不同的 GPU 组上，也支持同卡混合。 |
| slime（THUDM） | Megatron + SGLang 的 RL 框架，支持同步和异步，GLM-4.5 系列的 RL 训练使用了它（待核实细节）。 |
| AReaL（蚂蚁 + 清华） | 完全异步的 RL 系统：生成可中断、staleness 受控，配 decoupled PPO 目标。 |
| NeMo-RL（NVIDIA）、ROLL（阿里）、TRL（HF） | 分别面向 NVIDIA 生态、大规模多任务、易用性（具体特性待核实）。 |

> [!Q]
> **如果让你从零设计一个 RL 训练系统，你先考虑什么？**
> ①模型规模和回复长度分布，决定 rollout 的占比和显存压力；②能接受多大的 off-policy，决定同步、one-step-off 还是 fully async；③据此选择 colocate 或分池；④权重同步的路径和频率；⑤长尾处理（partial rollout）；⑥reward 的计算方式（规则、沙箱、RM）以及是否能和生成重叠；⑦可观测性：每个阶段的耗时（verl 的 `timing_raw`）、staleness、训推 log-prob 差异。

> [!WARN]
> MoE 模型做 RL 时还有一个额外的不一致来源：推理引擎和训练框架的路由结果可能不同（数值差异让 top-k 选到不同专家），训推 log-prob 差异比 dense 模型更大。有工作把 rollout 时的路由结果记录下来、训练时按原路由计算（具体方案待核实）。
