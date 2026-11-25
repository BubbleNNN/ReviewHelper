---
title: verl 专项
summary: 架构、训练循环、配置与源码细节，按面试官的追问顺序组织。
tags: verl, RL
order: 50
---

## 用两三句话介绍 HybridFlow 的设计。

RL 训练是一张数据流图，节点内是大模型分布式计算，节点间是多对多数据重分布。HybridFlow 把**控制流**放在单进程 driver（写算法像写单机程序），把**计算**交给 Ray 管理的多控制器 SPMD worker 组（FSDP / Megatron / vLLM）。两者之间用 `@register(dispatch_mode=...)` 声明数据怎么切分和收集。

## single-controller 和 multi-controller 各有什么优缺点？

单控制器：一个中心进程编排所有计算，灵活、易写新算法，但每个分布式算子都要中心派发，开销大。多控制器：每卡一个进程跑同一份程序（SPMD），效率高，但控制流和并行实现耦合，换算法要改分布式代码。HybridFlow 在粗粒度用单控制器，细粒度用多控制器。

## `DP_COMPUTE_PROTO` 是什么意思？

一种 dispatch 模式：driver 传入的 `DataProto` 按 DP rank 数切分，每个 rank 处理一份，结果按顺序拼接回一个 `DataProto`。其他预定义模式还有 `ONE_TO_ALL`（广播同一参数，如 `init_model`）、`RANK_ZERO`、`ALL_TO_ALL`、`DP_COMPUTE_METRIC` 等。

## 为什么 actor 和 rollout 要共置在同一批 GPU 上？

RL 的生成和训练是串行依赖的，分开部署会让一边空转。共置后两者时分复用显存：rollout 时装推理权重和 KV cache，训练时 sleep 推理引擎、装训练状态。权重同步在本地完成（CUDA IPC / 本地重切），比跨机传输快得多。

## 3D-HybridEngine 解决什么问题？

训练和生成的并行布局不同（如训练 FSDP 全分片或 Megatron TP×PP×DP，生成 vLLM TP=2）。每次训练完要把参数重切成生成布局。3D-HybridEngine 让这个重切不保留冗余副本、通信量尽量小。现在 verl 把这部分抽象为 checkpoint engine，支持 naive / NCCL / NIXL / Mooncake 等后端。

## V1 trainer 的一个 step 有哪些阶段？顺序是？

采样 batch（gen）→ 奖励（reward）→ DP 负载均衡 → old log-prob → ref log-prob（可选）→ values（可选）→ 优势（adv）→ 更新 critic（可选）→ 更新 actor（过了 `critic_warmup` 才更新）→ 权重同步。

## `train_batch_size=256, rollout.n=8, ppo_mini_batch_size=64, ppo_epochs=2`，一个 step 做几次 optimizer step？每个 mini-batch 多少条 response？

$256/64\times2=8$ 次 optimizer step。`ppo_mini_batch_size` 的单位是 prompt，内部乘以 `rollout.n`，所以每个 mini-batch 有 $64\times8=512$ 条 response（全局）。

## `ppo_micro_batch_size_per_gpu` 改了会不会影响训练结果？

理论上不会，它只决定梯度累积时每次前向/反向的样本数，影响显存和速度。实际有微小数值差异（求和顺序不同）。真正影响算法的是 `ppo_mini_batch_size`（每次更新的数据量）和 `ppo_epochs`。

## 为什么要重新计算 old_log_prob，而不直接用推理引擎返回的 log-prob？

推理引擎和训练引擎的数值实现不同（kernel、精度、batch 不变性），log-prob 会有偏差；用训练引擎重算保证 PPO ratio 的基准一致，第一个 mini-batch 的 ratio 精确为 1。偏差本身可以用 rollout correction（TIS / 拒绝采样）处理；`bypass_mode` 则直接用 rollout log-prob 省掉这次前向。

## verl 里 KL 可以放在哪两处？默认用什么估计量？

放 reward：`algorithm.use_kl_in_reward`，默认估计量 `kl`（k1），`kl_ctrl` 默认 fixed、系数 0.001。放 loss：`actor.use_kl_loss`，默认 `low_var_kl`（k3），系数 0.001。GRPO 系一般放 loss。只要开了其中任意一个，就需要 ref 模型；两个都关时不建 ref。

## k3 估计量是什么？为什么叫 low variance？

记 $r=\pi_{\text{ref}}/\pi$，$k_3=(r-1)-\log r$。在 $\pi$ 下采样时 $\mathbb E[r-1]=0$，所以 $\mathbb E[k_3]=\mathbb E[-\log r]=\mathrm{KL}(\pi\|\pi_{\text{ref}})$，无偏；它恒非负，且 $r-1$ 是一个与 $-\log r$ 负相关的控制变量，方差比 k1 小。verl 还会把它 clamp 到 $[-10,10]$。

## `kl_loss_type=k3+` 中的 `+` 是什么？

straight-through：前向数值用 k3，反向梯度用 k2（$\tfrac12(\log\pi-\log\pi_{\text{ref}})^2$）的梯度。因为 k1/k3 的期望等于 KL，但它们梯度的期望不等于 KL 的梯度；k2 的梯度期望正确。

## 说说 `loss_agg_mode` 的几种取值和区别。

`token-mean`（默认）：全局 token 平均，长回答贡献更多 token。`seq-mean-token-mean`：先序列内平均再序列间平均（原 GRPO），长序列里每个 token 被稀释，长 CoT 下不稳定。`seq-mean-token-sum`：序列内求和再平均。`seq-mean-token-sum-norm`：再除以常数（Dr.GRPO），去掉长度归一化偏置。另有 `token-sum`。

## `token-mean` 在多卡下怎么保证和单卡一致？

各卡算本地 masked sum，除以**全局**有效 token 数 `batch_num_tokens`，再乘 `dp_size`；因为 DDP/FSDP 的梯度同步是取平均，乘回 `dp_size` 后等价于全局 token 平均。如果各卡自己除以本地 token 数，就等于给每张卡等权，和单卡不一致。

## verl 的 PPO 损失里 `clip_ratio_c` 起什么作用？

dual-clip。对 $A<0$ 的 token，把损失上界截在 $-A\cdot c$（默认 $c=3$），防止新策略大幅提高了坏动作概率时 $-A\rho$ 无限增长。要求 $c>1$。监控指标 `pg_clipfrac_lower` 就是被它截住的比例。

## 怎么让 verl 支持一个新的优势估计方法？

写函数并用 `@register_adv_est("name")` 注册，签名对齐现有函数（`token_level_rewards`、`response_mask`、`index`、`config`…），配置里设 `algorithm.adv_estimator=name`。策略损失同理用 `@register_policy_loss`。不用改 trainer。

## remove padding、dynamic bsz、balance batch 分别解决什么？

remove padding：消掉 micro-batch 内的 padding（varlen flash-attn）。dynamic bsz：按 token 预算装箱，让每个 micro-batch 的 token 数接近，显存可控。balance batch：在 driver 上按估算的计算量（$24576s+s^2$）用 Karmarkar–Karp 把样本分给 DP rank，减少 rank 间互相等待。

## 多轮 agent RL 为什么要 token-in/token-out？

解码再重新分词不可逆，token 序列可能变化，训练时算 log-prob 的序列就不是模型实际采样的序列，ratio 和梯度都错。所以每轮都把生成的 token id 原样拼接，工具返回的部分单独 tokenize 后拼上，并用 `response_mask=0` 标记。

## 工具返回的 token 在训练里怎么处理？

`response_mask=0`：不算 policy loss、不算熵和 KL；GAE 递推时跳过（value 和优势原样穿过）；但保留在输入里作为上下文。

## verl 的异步训练怎么控制策略滞后？

每条轨迹记录生成时的权重版本；replay buffer 采样时，滞后超过 `max_off_policy_threshold`（默认 8）的按策略丢弃；再配合 rollout correction 的重要性加权/拒绝采样。`separate_async` 每 `parameter_sync_step` 个 mini-batch 同步一次权重，期间用同一个 $\pi_{\text{old}}$（Decoupled PPO）。

## GRPO 一个 batch 里有很多全对或全错的组，有什么影响，怎么办？

这些组优势恒为 0，没有梯度，有效 batch 变小、梯度方差变大。办法：DAPO 的动态采样（`algorithm.filter_groups.enable`）过滤掉再补采；提高 `rollout.n`；做课程学习，筛掉太简单或太难的题。

## 如果 rollout 阶段特别慢，你会从哪些地方入手？

先看是不是被长尾拖住：限制 `max_response_length`、开 partial rollout / 异步。再看推理吞吐：调 `gpu_memory_utilization`、`max_num_seqs`、推理 TP，确认 CUDA Graph 开着（`enforce_eager=false`），开 prefix caching（GRPO 的 n 条共享 prompt）。最后看调度：agent 场景的粘性路由和负载均衡。
