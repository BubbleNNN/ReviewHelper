---
title: Static vs Continuous Batching 模拟器
difficulty: 中
tags: 推理调度, vLLM, Orca
order: 22
source: Orca (OSDI'22) iteration-level scheduling · vLLM scheduler
---

用离散的 step（一次模型前向）模拟推理引擎的调度。每个请求是 `(arrival_step, prompt_len, output_len)`，`output_len >= 1`。

**通用约定**

- 请求的 **prefill** 发生在它被调度的那个 step，这一步处理 `prompt_len` 个 token，同时产出**第 1 个**输出 token。之后每个 step 做一次 decode，处理 1 个 token、产出 1 个输出 token。所以请求的完成 step 是 `start + output_len - 1`。
- 每个 step 处理的 token 数（prefill 的 prompt token + decode token）不能超过 `token_budget`；同时在跑的序列数不能超过 `max_batch`。
- 调度严格 **FCFS**：按 `(arrival, 下标)` 排队，**队头放不下就停**，不能跳过队头去调度后面的请求。
- `token_budget < max_batch` 或某个 `prompt_len > token_budget` 时抛 `ValueError`。

**Static batching**（`simulate_static`）：引擎空闲时，从已到达的队列里按 FCFS 取一批（条数 ≤ `max_batch`，prompt 总和 ≤ `token_budget`），一起 prefill，然后一起 decode。**整批里最长的请求结束后**，下一批才能开始。已经结束的请求不再计入 token 数。

**Continuous batching**（`simulate_continuous`，iteration-level scheduling）：每个 step：

1. 先给所有在跑的序列各留 1 个 decode token 的预算。
2. 再按 FCFS 用剩余预算接纳新请求（做 prefill），直到队头没到达、batch 满了或者预算不够。
3. 本 step 结束时，完成的请求立刻离开，空出的位置下一个 step 就能用。

```python
simulate_static(requests, max_batch, token_budget) -> dict
simulate_continuous(requests, max_batch, token_budget) -> dict
# 返回:
#   "start":  每个请求被 prefill 的 step
#   "finish": 每个请求产出最后一个 token 的 step
#   "tokens_per_step":  每个 step 实际处理的 token 数（长度 = max(finish)+1，空闲 step 为 0）
#   "running_per_step": 每个 step 产出了 token 的序列数
#   "throughput": 总输出 token 数 / (max(finish)+1)
```

<!-- solution -->

### 讲解

**为什么 continuous batching 快那么多？** static batching 的问题有两个。一是短请求结束后，它的位置要空等到整批里最长的那个结束（大量 padding 计算）。二是新请求必须等整批结束才能进来，排队时间长。Orca 提出按 iteration（一次前向）粒度调度：每个 step 都可以有请求退出、有请求加入。vLLM、SGLang、TensorRT-LLM 都用这种方式。

**token budget 与 chunked prefill。** 一个长 prompt 的 prefill 会让这一步特别慢，同一 batch 里正在 decode 的请求被拖累（TPOT 抖动）。vLLM 的 `max_num_batched_tokens` 就是这里的 `token_budget`。开启 chunked prefill 后，长 prompt 会被切成几段，分几个 step 和 decode 混在一起跑。本题没有切分，prompt 必须一次放进预算。

**decode 优先。** 先给在跑的序列留预算，再接纳新请求，这样 decode 不会被饿死，TPOT 稳定。代价是高负载时新请求的 TTFT 变长。vLLM V1 的调度器也是先排 running，再排 waiting。

**易错点。**

- 被调度的那一步同时完成了 prefill 和第一个 token，`output_len == 1` 的请求当步就结束。
- 严格 FCFS 下，队头一个大 prompt 会挡住后面的小请求（head-of-line blocking）。实际系统会做优先级、抢占，或者用 chunked prefill 缓解。
- 统计吞吐时要把空闲的 step 算进分母。

> [!Q]
> TTFT 和 TPOT 怎么权衡？PD 分离解决的是什么？
>
> 同一个 GPU 上，prefill 是计算密集，decode 是访存密集，混在一起会互相干扰：prefill 让 TPOT 抖动，decode 占着 batch 让 TTFT 变长。PD 分离把 prefill 和 decode 放到不同的实例上，中间传 KV cache，两边可以按各自的 SLA 独立扩缩容和调参。

> [!Q]
> RL rollout 场景和在线服务的调度目标有什么不同？
>
> 在线服务看 SLA（TTFT/TPOT 的分位数）。RL rollout 看的是整批的完成时间：一个 step 要等所有样本生成完，所以长尾请求决定一切。常见做法是把 batch 开满追求吞吐，再用 partial rollout / 异步 RL 缓解长尾。
