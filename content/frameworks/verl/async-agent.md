---
title: Agent Loop 与异步训练
summary: 多轮工具调用的 rollout 怎么组织、为什么必须 token-in/token-out，以及 V1 的异步模式怎么处理长尾和策略滞后。
tags: verl, Agent, 异步 RL, 多轮
order: 50
---

## 为什么 rollout 要变成“服务”

单轮 RLVR 的 rollout 就是一次批量生成。Agent RL 不同：模型说一段话 → 调工具 → 拿到结果 → 再说，**每条轨迹的轮数、长度、工具耗时都不一样**。如果还按“整批同步生成”来做，一个慢工具就能卡住整个 batch。

verl 的做法是把推理引擎包成 server，每条轨迹由一个协程驱动：

<div class="flow"><span class="f-accent">AgentLoopManager<small>generate_sequences / wake_up / sleep</small></span><i>→</i><span>AgentLoopWorker × N<small>每个 worker 跑很多协程</small></span><i>→</i><span>AgentLoopBase.run()<small>一条轨迹的交互逻辑</small></span><i>→</i><span class="f-infra">LLMServerClient<small>首轮按负载路由，之后粘在同一 server</small></span><i>→</i><span class="f-infra">vLLM / SGLang server</span></div>

- `rollout.agent.num_workers`（默认 8）个 `AgentLoopWorker`，每个里面用 asyncio 并发跑大量轨迹。
- 默认 agent 是 `single_turn_agent`；多轮工具调用用 `tool_agent_loop`，配置在 `rollout.multi_turn`（如 `max_assistant_turns`、`max_tool_response_length`，默认 256）。
- **粘性会话**：同一条轨迹的后续轮次路由到同一个 server（按 request_id），这样它的前缀 KV cache 还在，能命中 prefix caching。

自定义 agent 只需实现一个方法：

```python
class MyAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        # 1. 把 prompt 变成 token ids
        # 2. 循环：调 server.generate(token_ids) -> 解析工具调用 -> 执行工具 -> 拼接
        # 3. 返回 prompt_ids / response_ids / response_mask ...
        ...
```

`AgentLoopOutput` 的关键字段：

| 字段 | 含义 |
|---|---|
| `prompt_ids` / `response_ids` | token 级别的输入与输出，response 里**同时包含**模型生成的 token 和工具返回的 token |
| `response_mask` | 1 = 模型生成，0 = 工具返回。训练时只对 1 的位置算 loss，GAE 也跳过 0 的位置 |
| `response_logprobs` | 推理引擎给出的 log-prob，用于 rollout correction |
| `routed_experts` | MoE 模型在 rollout 时每个 token 走了哪些专家，训练时可以回放同样的路由，减少训推不一致 |
| `reward_score`、`num_turns`、`metrics` | 奖励（可在 rollout 内算好）、轮数和性能统计 |

## Token-in / Token-out

> [!KEY]
> 多轮 rollout 的每一轮都要把**上一轮生成的 token id 原样拼回去**，不能“解码成文本 → 套 chat template → 重新 tokenize”。

原因：BPE 分词不是可逆的一一映射。模型生成的 token 序列 `["▁foo", "bar"]` 解码成 `"foobar"` 后，重新分词可能变成 `["▁foob", "ar"]`。这样训练时计算 log-prob 的序列和模型实际采样的序列不是同一个，等于在“模型没生成过的轨迹”上做策略梯度，ratio 和优势都错了，收敛会变差。verl 的 server 提供 token 进、token 出的 `generate` 接口，就是为了这个。配置里 `tokenization_sanity_check_mode: strict` 会检查拼接前后是否一致。

> [!Q]
> **工具返回的 token 要不要算 loss？要不要算 KL？**
> 都不要。它们不是策略生成的，`response_mask=0`，在 policy loss、entropy、KL 的聚合里都被 mask 掉；GAE 递推时也跳过（`nextvalues`、`lastgaelam` 原样穿过）。但它们要留在上下文里，因为后续 token 的条件概率依赖它们。

## 长尾与异步

同步训练的问题：一个 batch 要等**最长**的那条轨迹生成完，GPU 大部分时间在等少数长序列。V1 的异步模式用两个机制缓解：

1. **Partial rollout**：生成到一半的轨迹可以跨越权重更新继续生成。轨迹里不同段可能来自不同版本的策略。
2. **Replay buffer + 滞后控制**：生成好的轨迹进 buffer，trainer 从 buffer 采样。每条轨迹记录生成时的策略版本，滞后超过 `sampler.max_off_policy_threshold`（默认 8）的按 `max_off_policy_strategy`（默认 `drop`）处理。

| 模式 | 资源 | 何时同步权重 |
|---|---|---|
| `sync` | 训练、推理共置 | 每个 step 末尾 |
| `colocate_async` | 共置，开 partial rollout，有 `num_warmup_batches` 预热 | 每个 step 末尾 |
| `separate_async` | 训练、推理分池；`hybrid_rollout.enable_switch` 时训练侧空闲可切去 rollout | 每 `parameter_sync_step`（默认 4）个 mini-batch |

`separate_async` 里，一个 step 内做 `parameter_sync_step` 次本地更新，这几次都要用**同一个** $\pi_{\text{old}}$：trainer 把那份权重存在 CPU，算 old log-prob 时临时换回来（Decoupled PPO 的做法）。也可以开 `rollout_correction.bypass_mode`，直接用 rollout log-prob，省掉这次前向。

> [!Q]
> **异步 RL 带来什么问题？怎么修正？**
> 轨迹来自旧策略，变成了 off-policy。修正手段：(1) 限制滞后步数，超了就丢；(2) 重要性采样修正，用 $\pi_{\text{old}}/\pi_{\text{behavior}}$ 加权并截断（TIS），或者直接拒绝偏离大的样本；(3) Decoupled PPO：把“近端策略”（clip 的中心）和“行为策略”（生成数据的策略）分开，clip 以近端策略为中心，行为策略只出现在重要性权重里。

> [!Q]
> **训推分离时权重怎么同步？**
> 训练侧把参数从训练布局（FSDP 分片 / Megatron TP×PP）汇集成推理需要的完整张量或推理 TP 分片，再发给每个推理实例。verl 的 checkpoint engine 支持 NCCL/HCCL 广播、NIXL 和 Mooncake 的 ring p2p 等后端；流式发送可以避免把整个模型在一张卡上物化。另有 delta weight sync（只传变化量）等优化。

## 老方案：one-step-off 与 fully async

这两个早期异步方案现在在 `verl-recipe` 里，基于 V0 trainer，固定 v0.9.1：

- **one-step-off-policy**：rollout 和训练分池，第 $k$ 步训练时 rollout 已经在用第 $k-1$ 步的权重生成下一批，滞后恒为 1。
- **fully async**：生成和训练完全解耦，靠 staleness 阈值和 partial rollout 控制。

面试时可以把它们作为“V1 async 模式的前身”来讲，重点讲清楚滞后和修正。

> [!SRC]
> `verl/experimental/agent_loop/agent_loop.py`（`AgentLoopBase`、`AgentLoopOutput`、`AgentLoopWorker`）、`tool_agent_loop.py`；V1 async：`verl/trainer/ppo/v1/trainer_colocate_async.py`、`trainer_separate_async.py`、`replay_buffer.py`。
