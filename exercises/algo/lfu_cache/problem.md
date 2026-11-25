---
title: LFU 缓存
difficulty: 难
tags: 哈希表, 双向链表, 设计
order: 42
source: LeetCode 460 · 和 LRU 对比的缓存淘汰策略
---

设计一个容量为 `capacity` 的 LFU（最不经常使用）缓存，`get` 和 `put` 都要求 **O(1)**。

```python
cache = LFUCache(2)
cache.put(1, 1); cache.put(2, 2)
cache.get(1)      # 1；key 1 的使用次数变成 2
cache.put(3, 3)   # 满了：淘汰使用次数最少的 key 2
cache.get(2)      # -1
cache.get(3)      # 3
cache.put(4, 4)   # 1 和 3 的次数都是 2，淘汰其中最久没被使用的 1
cache.get(1)      # -1
```

- 每个 key 有一个使用次数：`put` 新 key 时为 1；之后每次 `get` 命中或 `put` 更新已有 key，次数 +1。
- 满了要插入新 key 时，先淘汰**次数最少**的；次数相同的，淘汰**最久没被使用**的（LRU）。
- `capacity == 0` 时 `put` 什么也不做。
- 实现 `__len__`。

<!-- solution -->

### 讲解

**结构**：三张表 + 一个变量。

- `key -> (value, freq)`；
- `freq -> 有序的 key 集合`：同一频次内部按最近使用排序。Python 里可以用 `OrderedDict` 当“带 O(1) 删除的双向链表”（这题允许用，LRU 那题不允许）；
- `min_freq`：当前最小频次。

**操作**：

- `_touch(key)`：从 `freq` 桶里删掉，`freq+1` 后追加到新桶末尾；如果旧桶空了且旧频次正好是 `min_freq`，`min_freq += 1`。
- `get`：命中就 `_touch`。
- `put` 已有 key：更新 value 并 `_touch`。
- `put` 新 key：满了就从 `min_freq` 桶的**头部**弹出一个（最久未用）删掉；插入新 key，`freq=1`，`min_freq = 1`。

每一步都是常数次哈希操作，$O(1)$。

**LRU vs LFU**：

| | LRU | LFU |
|---|---|---|
| 依据 | 最近一次访问时间 | 访问次数 |
| 优点 | 实现简单，适应访问模式变化快 | 能留住长期热点 |
| 缺点 | 一次大扫描会冲掉热点 | 历史热点不会“老化”，新热点进不来（缓存污染） |
| 改进 | LRU-K、2Q、ARC | 次数衰减、W-TinyLFU |

**易错点**：

- 插入新 key 后 `min_freq` 一定是 1，别忘了重置。
- `put` 更新已有 key 也算一次使用。
- `capacity == 0` 是常见的边界陷阱。

> [!Q]
> **推理引擎的 prefix cache 为什么多用 LRU 而不是 LFU？** 对话请求的前缀热度变化很快（一轮对话结束后很快不再被引用），LRU 更贴近“最近还会用到”的规律；而 system prompt 这类长期热点，因为一直有请求在引用，本身就不会被淘汰。LFU 的历史计数反而会让过期的热点长期占着显存。

> [!Q]
> **Caffeine 的 W-TinyLFU 是什么思路？** 用 Count-Min Sketch 以很小的内存近似记录访问频次并定期减半（老化），新元素先进一个小的 LRU 窗口，被挤出时和主缓存的淘汰候选比频次，频次高的才能进入主缓存。兼顾了 LRU 的时效性和 LFU 的抗扫描能力。
