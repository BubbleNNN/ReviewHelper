---
title: 序列长度均衡与动态 micro-batch
difficulty: 中
tags: 负载均衡, verl, 贪心
order: 18
source: verl/utils/seqlen_balancing.py · get_seqlen_balanced_partitions / rearrange_micro_batches
---

RL 的 rollout 长度差异极大（几十到几万 token）。如果按条数平均分给各个 DP rank，有的 rank 拿到一堆长序列，其他 rank 只能干等。verl 的 `balance_batch` 和 `use_dynamic_bsz` 就是解决这个问题的。实现三个函数。输入输出都是普通的 python `list[int]` / `int`，不需要 tensor。

**1. 均衡分组**：把 `seqlens` 分成 `k` 组，让**最大组的总长度尽量小**。`equal_size=True` 时每组条数相同（要求 `len(seqlens) % k == 0`，否则抛 `ValueError`）。可以用 LPT 贪心（按长度降序，每次放进当前总长最小、且没满的组）或 Karmarkar-Karp 差分法。

```python
balanced_partition(seqlens, k, equal_size=True) -> list[list[int]]   # 每组是原下标列表
```

**2. 按 token 预算切 micro-batch**：每个 micro-batch 的总 token 数不超过 `max_token_len`，micro-batch 个数尽量少（装箱问题，用 First-Fit Decreasing）。有单条序列超过预算时抛 `ValueError`。

```python
micro_batches_by_tokens(seqlens, max_token_len) -> list[list[int]]
```

**3. 还原顺序**：按 `groups` 的顺序依次跑完所有 micro-batch，把输出拼起来得到 `outputs`（第 `pos` 个输出对应 `order[pos]`，其中 `order` 是 groups 展平后的下标序列）。返回 `rev`，使得 `outputs[rev[i]]` 就是原第 `i` 条序列的输出。

```python
reverse_index(groups) -> list[int]
```

<!-- solution -->

### 讲解

**为什么要均衡？** 每个训练 step 所有 DP rank 要一起做梯度同步，step 时间由最慢的 rank 决定。按 token 总数（或 attention 的平方项）均衡后，长尾 rank 的等待时间大幅下降。verl 的 `_balance_batch` 先用 `calculate_workload` 把每条序列换算成近似计算量 $24576L + L^2$（按 7B、hidden 4096 标定，第二项对应 attention 的平方成本），再调用 `get_seqlen_balanced_partitions(..., k_partitions=dp_size, equal_size=True)`。这个函数用的是 Karmarkar-Karp 差分法，分好的组按 DP rank 顺序重排整个 batch。

**LPT 的保证。** 不限条数时，LPT 的最大负载不超过 $\frac{4}{3}-\frac{1}{3k}$ 倍最优。更粗但好记的界是 $\le \frac{\sum L}{k} + \max L$（list scheduling 的通用界）。

**FFD 的保证。** First-Fit Decreasing 用的箱子数 $\le \frac{11}{9}\text{OPT} + \frac{6}{9}$。任意 First-Fit 类算法最多只有一个箱子不到半满，所以箱子数 $\le 2\sum L / C + 1$。

**动态 batch 的意义。** `use_dynamic_bsz=True` 时，verl 不再固定每个 micro-batch 的条数，而是按 `ppo_max_token_len_per_gpu` 打包。长序列少放几条，短序列多放几条，显存占用平稳，还减少了 padding。通常配合 `use_remove_padding` 一起用：把序列首尾拼成一维（varlen / packing），不同长度的序列放在一个 micro-batch 里也不用补 padding。

**易错点。**

- 打乱顺序之后一定要能还原，否则 advantage、log-prob 会和样本错位。这个 bug 很隐蔽，loss 看起来照样能降。
- 每个 DP rank 的 micro-batch 数要一致（`same_micro_num_in_dp`），否则 FSDP/Megatron 的集合通信会卡死。
- 梯度累积按 token 平均时，要用全局 token 数归一化，不能对每个 micro-batch 先取平均。

> [!Q]
> 动态 batch 会改变训练的数学吗？
>
> 不应该。micro-batch 只是梯度累积的切法，mini-batch（一次 optimizer step 的样本集合）没变。只要 loss 按全局 token 数或全局样本数归一化（verl 的 `agg_loss` 传入 `batch_num_tokens` / `global_batch_size`），结果就和不切一样。

> [!Q]
> GRPO 的一组 rollout 要不要放在同一个 rank？
>
> 优势在 driver 上已经按组算完，训练时样本之间独立，所以拆开没问题。但如果在 worker 上算组内统计，或者想利用同组共享的 prompt 前缀，就需要把同组放在一起。verl 的 `rearrange_micro_batches` 有一个 `force_group_size` 参数，强制连续若干条样本落在同一个 micro-batch，源码注释写的用途是 RM 训练。
