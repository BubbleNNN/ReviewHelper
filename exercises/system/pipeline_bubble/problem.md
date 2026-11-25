---
title: 流水线并行：GPipe 与 1F1B 调度模拟
difficulty: 中
tags: 流水线并行, Megatron, 调度
order: 16
source: GPipe (2019) · PipeDream-Flush / Megatron-LM 1F1B (2021)
---

模拟 $p$ 个 stage、$m$ 个 micro-batch 的流水线训练。每个 micro-batch 在每个 stage 上前向耗时 `tf`、反向耗时 `tb`，忽略通信。依赖关系：

- `F(s, i)` 必须等 `F(s-1, i)` 完成（$s>0$）
- `B(s, i)` 必须等 `F(s, i)` 完成，并且等 `B(s+1, i)` 完成（$s<p-1$）
- 每个 stage 同一时刻只做一件事，**严格按照给定顺序**执行自己的操作，能开始就立刻开始

两种调度的每个 stage 的操作顺序：

- **GPipe**：`F0, F1, …, F(m-1)`，然后 `B0, B1, …, B(m-1)`
- **1F1B**（非交错版）：stage $s$ 先做 $w=\min(p-s-1,\ m)$ 个前向（warmup），然后交替“一前一后” `F(w+i), B(i)`（steady），最后把剩下的反向做完（cooldown）

```python
simulate(schedule, p, m, tf=1.0, tb=2.0) -> dict
# schedule ∈ {"gpipe", "1f1b"}
# 返回:
#   "timeline":      长度 p 的 list，每个元素是该 stage 上 (op, mb, start, end) 的列表，op ∈ {"F","B"}
#   "total_time":    所有 stage 的最晚结束时间
#   "bubble_ratio":  1 - m*(tf+tb) / total_time   （每个 stage 的空闲比例）
#   "peak_inflight": 长度 p 的 list，每个 stage 同时持有前向激活的 micro-batch 数的峰值
#                    （按该 stage 的操作顺序：做一次 F 加 1，做一次 B 减 1）
```

<!-- solution -->

### 讲解

**实现。** 写一个“谁能动谁就动”的离散事件循环：每个 stage 维护一个指针指向下一个操作，依赖都完成了就安排它：`start = max(stage 空闲时刻, 依赖的结束时刻)`。反复扫描所有 stage，直到全部排完。如果某一轮谁都动不了，说明调度死锁了。

**结论。** 两种调度的总时间都是 $(m+p-1)(t_f+t_b)$，bubble 比例都是

$$\text{bubble} = \frac{p-1}{m+p-1}$$

1F1B **并没有减少 bubble**。它的好处在显存：GPipe 每个 stage 要同时存 $m$ 份激活，1F1B 的 stage $s$ 最多存 $\min(p-s,\ m)$ 份，和 $m$ 无关。于是可以把 $m$ 开得很大来摊薄 bubble。

**怎么继续减少 bubble？**

- 增大 $m$（受 global batch 限制）
- **interleaved 1F1B**（virtual pipeline）：每张卡放 $v$ 段不连续的层，bubble 降到约 $\frac{1}{v}\cdot\frac{p-1}{m}$，代价是通信次数变成 $v$ 倍
- zero-bubble / DualPipe 一类方法：把反向拆成对输入的梯度和对权重的梯度两部分，用权重梯度去填空隙

**易错点。** warmup 个数是 $p-s-1$ 而不是 $p-s$：最后一个 stage 不做 warmup，一上来就是 F0 紧接 B0。$m<p$ 时 warmup 要截断到 $m$。

> [!Q]
> PP 和 TP 怎么搭配？为什么 PP 常用于跨节点？
>
> PP 只在相邻 stage 之间点对点传激活，通信量小、对带宽不敏感，适合跨节点。TP 每层都要 all-reduce，必须放在 NVLink 域内。典型配置是节点内 TP=8，节点间 PP，再外面一层 DP。

> [!Q]
> RL 训练里为什么很多框架（比如 verl 的 FSDP 后端）宁可不用 PP？
>
> PP 需要足够多的 micro-batch 才能摊薄 bubble，还会和 rollout 引擎之间的权重重分片（resharding）纠缠在一起，实现复杂。中等规模下 FSDP + SP 更简单；超大模型才上 Megatron 的 TP/PP/EP。
