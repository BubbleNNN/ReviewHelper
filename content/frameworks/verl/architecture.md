---
title: HybridFlow 架构：单控制器管流程，多控制器管计算
summary: verl 为什么这样分层、Worker / ResourcePool / Dispatch 这几个抽象各管什么，以及训练和推理怎么共用一批 GPU。
tags: verl, HybridFlow, Ray, 架构
order: 10
---

## verl 是什么

verl 是字节跳动 Seed 团队开源的 LLM 强化学习后训练框架，论文是 **HybridFlow: A Flexible and Efficient RLHF Framework**（arXiv 2409.19256）。仓库 2026 年 1 月迁到了 `verl-project` 组织下，本页对照的源码版本是 `0.10.0.dev`（2026-11）。

一句话概括：**用一个单进程的 driver 写 RL 算法的控制流，用 Ray 管理的多组 SPMD worker 跑分布式的生成、打分和训练**。

<div class="flow"><span class="f-accent">Driver（单控制器）<small>trainer.fit()：算法逻辑</small></span><i>→</i><span class="f-infra">Rollout WorkerGroup<small>vLLM / SGLang 生成</small></span><i>→</i><span>Ref / Critic / RM<small>前向打分</small></span><i>→</i><span class="f-rl">Actor / Critic 训练<small>FSDP / Megatron</small></span><i>↺</i><span class="f-infra">权重同步<small>checkpoint engine</small></span></div>

## RL 是一张数据流图

RLHF / RLVR 的一个 step 可以画成一张 DAG：节点是“分布式程序”（生成、算 log-prob、算 value、更新参数），边是节点之间的数据依赖。难点在两处：

1. **节点内**是大模型计算，本身就要 TP / PP / DP / FSDP 等并行策略；
2. **节点间**的数据要“多对多重分布”：rollout 用 TP=2 生成出来的数据，要按 FSDP 的 DP 维度切给训练进程。

| 方案 | 怎么组织 | 优点 | 缺点 |
|---|---|---|---|
| 单控制器（single-controller） | 一个中心进程调度所有设备上的计算 | 写算法像写单机程序，灵活 | 每个算子都经过中心调度，分布式计算的派发开销大 |
| 多控制器（multi-controller / SPMD） | 每张卡一个进程跑同一份程序，靠集合通信协作 | 计算效率高，Megatron / FSDP 都是这个范式 | 控制流和计算嵌在一起，换个算法就要改分布式代码 |
| **HybridFlow** | 控制流放单控制器，节点内计算交给多控制器 | 两者兼得 | 需要一套“数据怎么分发给 worker”的协议 |

> [!KEY]
> HybridFlow 的核心取舍：**粗粒度**的东西（调用哪个模型、按什么顺序、算什么优势）由 driver 决定；**细粒度**的东西（矩阵怎么切、梯度怎么同步）留在各 worker 组内部用 SPMD 完成。driver 只在 RPC 边界上搬数据。

## 四个核心抽象

### ResourcePool 与 WorkerGroup

- `ResourcePool`：Ray 集群里的一组 GPU（例如 2 节点 × 8 卡）。
- `RayWorkerGroup`：在一个 ResourcePool 上**每张卡起一个 Ray actor**，对 driver 暴露成一个“代理对象”。driver 调 `wg.compute_log_prob(batch)`，WorkerGroup 负责把 batch 切开发给每个 rank、收集结果。
- 多个角色可以**共置（colocate）**在同一个 ResourcePool 上：`create_colocated_worker_cls` 把多个 worker 类融合进同一个进程，比如 Actor + Rollout + Ref 合成一个 `ActorRolloutRefWorker`。

### `@register` 与 Dispatch 模式

Worker 的方法用装饰器声明“数据怎么分发、结果怎么收集”：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_log_prob(self, data: DataProto) -> DataProto:
    ...
```

源码 `verl/single_controller/base/decorator.py` 里预定义的模式：

| Dispatch | 含义 |
|---|---|
| `RANK_ZERO` | 只在 rank 0 执行 |
| `ONE_TO_ALL` | 同一份参数广播给所有 rank（如 `init_model`） |
| `ALL_TO_ALL` | 参数原样分发、结果原样返回 |
| `DP_COMPUTE` / `DP_COMPUTE_PROTO` | 按 DP 维度切分输入，各 rank 计算后再拼接；`_PROTO` 版要求输入输出是 `DataProto` |
| `DP_COMPUTE_PROTO_WITH_FUNC` | 同上，额外传一个函数 |
| `DP_COMPUTE_METRIC` | DP 计算，结果按 metric 收集 |
| `DIRECT_ROLLOUT_METHOD` | 给 vLLM 外部执行器用的特殊模式 |

另有 `Execute` 枚举（`ALL` / `RANK_ZERO`）决定在哪些 rank 上真正执行。

> [!WARN]
> 老文档和不少博客里写的 `MEGATRON_COMPUTE_PROTO` 在当前源码的预定义列表里**已经没有了**。面试时说“按 DP 切分的 `DP_COMPUTE_PROTO`”最稳。

### DataProto 与 TransferQueue

- `DataProto`（`verl/protocol.py`）= `batch: TensorDict` + `non_tensor_batch: dict`（numpy 对象数组，如 uid、原始文本）+ `meta_info: dict`。支持 `split` / `chunk` / `concat` / `union` / `repeat`，是 driver 和 worker 之间的标准数据格式。
- V0 时代所有 batch 都要经过 driver 中转，数据大时 driver 成为瓶颈。V1 trainer 引入 **TransferQueue**（Ascend 开源，2026-04 合入）：数据放在分布式 KV 存储里，driver 只传元数据 `KVBatchMeta`，worker 按 key 直接取。

## 共置：训练和推理共用一批 GPU

最常见的部署是 **hybrid engine**（配置 `actor_rollout_ref.hybrid_engine: true`）：同一批 GPU 上既有 FSDP/Megatron 训练状态，又有 vLLM/SGLang 推理引擎，二者**时分复用**。

<div class="flow"><span class="f-infra">wake_up<small>同步新权重</small></span><i>→</i><span class="f-infra">rollout 生成<small>KV cache 占显存</small></span><i>→</i><span>sleep<small>释放 KV cache、卸载推理权重</small></span><i>→</i><span class="f-rl">训练<small>参数 / 梯度 / 优化器状态</small></span><i>↺</i></div>

为什么值得这么做：

- RL 里 rollout 往往占一个 step 的大头时间，训练和推理分开部署会让一边空转；
- actor 与 rollout 在同一张卡上，权重同步走本地拷贝 / CUDA IPC，最快；
- actor 与 ref 共置在 LoRA 场景下尤其划算：ref 就是关掉 adapter 的 base model，可以共享权重。

### 3D-HybridEngine：两种并行布局之间重切权重

训练用的切法（例如 FSDP 全分片，或 Megatron TP×PP×DP）和推理用的切法（例如 vLLM TP=2）不同。HybridFlow 论文提出 **3D-HybridEngine**，在两种布局间重切参数，做到推理阶段**不保留冗余副本**、通信量尽量小。

现在这部分抽象成了 **checkpoint engine**（`verl/checkpoint_engine/`），统一了三个 API：`send_weights`（训练侧流式发出命名张量）、`receive_weights`（推理侧流式接收）、`get_weights`（推理实例从本地缓存独立加载）。

| backend | 通信 | 适用场景 |
|---|---|---|
| `naive` | `torch.distributed` all_gather | 同步训练，actor/rollout 共置 |
| `nccl` / `hccl` | all_gather + broadcast | 训推分离、固定集群 |
| `nixl` | all_gather + ring p2p | 训推分离、弹性 rollout、异构硬件 |
| `mooncake` | Mooncake Transfer Engine | 训推分离，可动态调整 ring 拓扑 |
| `kimi_ckpt_engine` | Mooncake + NCCL/HCCL | 训推分离，p2p 到一个 worker 再广播 |

> [!SRC]
> `verl/checkpoint_engine/README.md` 有完整的表和 benchmark；V1 trainer 的 `on_step_end` 调 `checkpoint_manager.update_weights(global_steps)` 完成同步。

## Worker 角色与后端

| 角色 | 做什么 | 后端 |
|---|---|---|
| Actor | 策略模型训练、算 old log-prob | FSDP / FSDP2、Megatron、TorchTitan、VeOmni、Automodel（`verl/workers/engine/`） |
| Rollout | 生成 response | vLLM、SGLang、TensorRT-LLM、HF（`rollout.name`） |
| Ref | 算参考策略 log-prob（KL 用） | 同 Actor，常开 param offload |
| Critic | 价值模型（只有 GAE 需要） | 同 Actor |
| Reward | 规则函数 `compute_score` 或奖励模型；V1 里由 RewardLoop 管理 | 函数 / 序列分类模型 |

> [!Q]
> **为什么不把 rollout 也写成 SPMD，直接嵌进训练进程？**
> 推理引擎（vLLM / SGLang）有自己的调度器、KV cache 管理和 CUDA Graph，生命周期和训练不同；agent 场景还需要异步、多轮、工具调用。把它包成独立的 server（`AgentLoopManager` + `LLMServerClient`）更容易做异步和负载均衡。verl 现在的 rollout 默认就是 server 模式（`rollout.mode: async`）。

> [!Q]
> **单控制器会不会成为瓶颈？**
> 会，主要是数据搬运：每个 step 的 batch 都经过 driver 序列化。缓解办法：(1) 只传必要字段；(2) V1 用 TransferQueue 让数据面绕过 driver，driver 只管控制面；(3) 计算本身一直在 worker 内部完成，driver 不碰张量运算。

> [!Q]
> **colocate 和 disaggregate（训推分离）各适合什么？**
> 同步 on-policy 训练用 colocate，资源利用率高、权重同步快；需要异步（rollout 和训练重叠）或 rollout 长尾严重时用分离部署，代价是权重要跨机同步、要处理 staleness。verl V1 的 `separate_async` 还允许训练侧空闲时临时切去做 rollout（`hybrid_rollout.enable_switch`）。
