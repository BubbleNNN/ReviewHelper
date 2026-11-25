---
title: 合并区间与插入区间
difficulty: 中
tags: 排序, 区间, 扫描
order: 22
source: LeetCode 56 / 57 · 显存空闲段合并
---

区间都是**闭区间** `[start, end]`，整数端点，`start <= end`。实现两个函数：

```python
merge(intervals) -> list[list[int]]
insert(intervals, new) -> list[list[int]]
```

- `merge`：输入**无序**，可能重叠、包含、首尾相接。合并所有有交集的区间，返回按 start 升序的结果。首尾相接也算有交集：`[1,3]` 和 `[3,5]` 合并成 `[1,5]`；但 `[1,2]` 和 `[3,4]` 之间隔着 (2,3)，**不合并**。
- `insert`：`intervals` 已经按 start 升序且互不重叠，插入 `new` 后返回同样规范的结果。要求 $O(n)$，不要直接调用排序。
- 不要修改输入。

```python
merge([[1, 3], [8, 10], [2, 6], [15, 18]])   # [[1, 6], [8, 10], [15, 18]]
insert([[1, 2], [3, 5], [6, 7], [8, 10]], [4, 8])   # [[1, 2], [3, 10]]
```

场景：显存/KV 块分配器回收一段空间时，要把它和左右相邻的空闲段合并，避免碎片化。`insert` 就是“释放一段，再合并”的那一步。

<!-- solution -->

### 讲解

**merge**：按 start 排序，然后线性扫描。维护结果里最后一个区间 `last`：

- 如果 `cur.start <= last.end`，有交集，`last.end = max(last.end, cur.end)`；
- 否则 `cur` 另起一段。

注意取 `max`：`[1,10]` 后面来 `[2,3]`，被包含的区间不能把 end 缩小。

**insert**：已有区间分三段处理，一次扫描：

1. 完全在 `new` 左边（`end < new.start`）的原样输出；
2. 和 `new` 有交集的（`start <= new.end`）全部吸收进 `new`：`new = [min(starts), max(ends)]`；
3. 输出 `new`，再原样输出右边剩下的。

**复杂度**：merge $O(n\log n)$（排序主导），insert $O(n)$。

**易错点**：

- 边界用 `<=` 还是 `<`，取决于闭区间还是半开区间。半开区间 `[s, e)` 下 `[1,3)` 和 `[3,5)` 是否合并要看业务：内存段“首尾相接”通常应该合并。
- 原地修改输入 list 里的子 list（`last[1] = ...`）会改到调用方的数据，要先复制。

> [!Q]
> **如果区间是动态增删的（不停 malloc/free），每次都重新排序太慢，怎么办？** 用按 start 排序的平衡树（C++ `std::map`、Python `sortedcontainers`），插入时只看前驱和后继两个邻居做合并，单次 $O(\log n)$。很多 allocator 的 free list 就是这么维护的。

> [!Q]
> **要回答“某个点被多少个区间覆盖”或“最大重叠数”呢？** 换成扫描线：把 start 记 +1、end 记 -1，排序后前缀和；或者用差分数组 / 线段树。见 meeting_rooms 那题。
