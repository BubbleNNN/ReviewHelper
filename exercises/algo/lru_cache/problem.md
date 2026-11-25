---
title: LRU 缓存
difficulty: 中
tags: 哈希表, 链表, 设计
order: 10
source: LeetCode 146 · 推理系统里 prefix cache 的淘汰策略
---

设计一个容量为 `capacity` 的 LRU（最近最少使用）缓存，`get` 和 `put` 都要求 **O(1)**。

```python
cache = LRUCache(2)
cache.put(1, 1); cache.put(2, 2)
cache.get(1)      # 1，并且 1 变成“最近使用”
cache.put(3, 3)   # 容量满，淘汰最久没用的 2
cache.get(2)      # -1
```

- `get(key)`：存在返回 value，否则返回 `-1`。命中会刷新为最近使用。
- `put(key, value)`：写入或更新，更新也算一次使用。超出容量时淘汰最久未使用的 key。
- 额外实现 `__len__`，返回当前元素个数。

**要求**：不要用 `collections.OrderedDict`，自己写哈希表 + 双向链表。

<!-- solution -->

### 讲解

- **结构**：`dict[key] -> node`，再用一条带哨兵头尾的双向链表维护使用顺序。头部是最近使用，尾部是最久未使用。
- **两个原子操作**：`_remove(node)` 和 `_add_front(node)`，都是改 4 个指针，O(1)。`get` 命中就是 remove + add_front；`put` 满了就删 `tail.prev`。
- **哨兵节点**：省掉所有“链表为空”“删的是头”之类的边界判断，面试时写起来最稳。
- **和 Infra 的联系**：vLLM 的 prefix caching 中，引用计数为 0 的 KV block 进入一个可淘汰队列，按 LRU 顺序回收；SGLang 的 RadixAttention 对 radix tree 的叶子节点做 LRU 淘汰。面试官常从这道题延伸到“KV cache 满了怎么办”。
