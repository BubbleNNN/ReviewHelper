---
title: 分割数组的最大值（流水线切分）
difficulty: 难
tags: 二分答案, 贪心, 动态规划
order: 30
source: LeetCode 410 · 流水线并行的 stage 划分
---

给定非负整数数组 `nums` 和整数 `k`（`1 <= k <= len(nums)`），把 `nums` 切成 `k` 个**非空、连续**的子数组，使各段和的**最大值最小**。

```python
split_array(nums, k) -> int              # 最优的“最大段和”
partition(nums, k) -> list[list[int]]    # 一种达到最优值的切法
split_array([7, 2, 5, 10, 8], 2)         # 18
partition([7, 2, 5, 10, 8], 2)           # [[7, 2, 5], [10, 8]]（最优切法可能不唯一）
```

`partition` 必须返回恰好 `k` 段、每段非空、拼起来等于 `nums`，并且最大段和等于 `split_array` 的结果。测试只检查这些性质。

场景：流水线并行（PP）要把 L 层 transformer 切成 p 个 stage，每层的耗时是 `nums[i]`。整个流水线的吞吐由最慢的 stage 决定，所以要让最大的 stage 耗时最小，就是这道题。

<!-- solution -->

### 讲解

**二分答案**：最大段和 `cap` 越大，需要的段数越少，具有单调性。在 `[max(nums), sum(nums)]` 上二分最小的可行 `cap`。

判定函数 `need(cap)`：从左往右贪心，当前段加上下一个数会超过 `cap` 就新开一段，返回段数。`need(cap) <= k` 就可行（段数不够 k 可以把某段拆开，拆开不会让最大值变大）。

**复杂度**：$O(n\log S)$，$S = \sum nums$。DP 解法是 $O(n^2 k)$：`dp[i][j]` = 前 i 个数切成 j 段的最优值。

**构造恰好 k 段**：拿到最优 `cap` 后再贪心一遍。为了保证段数恰好是 k，还要加一条“强制切”规则：如果剩下的元素个数刚好等于还需要的段数，就必须立刻切，之后每个元素单独成段。单个元素不超过 `max(nums) <= cap`，所以不会破坏最优性。

**易错点**：

- 二分下界是 `max(nums)` 而不是 0，否则单个大元素放不下。
- 贪心判定里是“超过 cap 才新开段”，等于 cap 时不用开。
- 有 0 的时候段可能和为 0，但仍然必须非空。

> [!Q]
> **真实的 PP 划分比这道题复杂在哪？** 第一个 stage 有 embedding、最后一个 stage 有 LM head 和 loss，耗时和显存都不均匀；还要同时考虑每个 stage 的**显存**上限（激活随 micro-batch 数累积）和 stage 之间的通信量。Megatron 允许手动指定每个 stage 的层数（如首尾 stage 少放几层）。

> [!Q]
> **为什么 PP 的吞吐由最慢 stage 决定？** 稳态下每个 micro-batch 都要依次经过所有 stage，流水线的节拍等于最慢 stage 的耗时，其他 stage 每拍都要等它，闲置时间就是浪费。
