---
title: 前缀缓存：Radix Tree + LRU 淘汰
difficulty: 中
tags: KV cache, SGLang, 前缀树
order: 24
source: SGLang RadixAttention (Zheng et al., 2024) · vLLM Automatic Prefix Caching
---

实现一个**按 token 建树**的前缀缓存（为了简单，每个节点只存一个 token；真正的 radix tree 会把单链压缩成一个节点）。节点代表“这个前缀的 KV 已经算过、存着”。

```python
c = PrefixCache()
c.insert(tokens) -> int          # 插入一条序列，返回新增的节点数
c.match_prefix(tokens) -> int    # 已缓存的最长前缀长度
c.lock(tokens)                   # 路径上每个节点 ref += 1（请求正在用，不可淘汰）
c.unlock(tokens)                 # 路径上每个节点 ref -= 1
c.evict(n_tokens) -> int         # 淘汰至少 n_tokens 个节点（或淘汰光可淘汰的），返回实际淘汰数
c.size() -> int                  # 当前节点总数
```

规则：

- **逻辑时钟**：每次调用 `insert` / `match_prefix` / `lock`，时钟 +1，并把这次访问路径上所有节点的最近访问时间设为当前时钟。
- **淘汰**：只能淘汰**叶子**且 `ref == 0` 的节点。每次挑最近访问时间最小的叶子删掉；删掉后父节点如果变成了叶子，且 `ref == 0`，也成为候选。重复直到淘汰数 ≥ `n_tokens` 或没有候选。
- `lock` 的序列必须已经完整缓存，否则抛 `ValueError`；`unlock` 一条没被锁住（路径上有 `ref == 0`）的序列也抛 `ValueError`。

<!-- solution -->

### 讲解

**为什么只淘汰叶子？** 子节点的 KV 依赖父节点的前缀。删掉中间节点，它下面的子树就没法用了。从叶子往上剥，任何时刻缓存里都是若干条完整的前缀。

**为什么父节点的访问时间不会早于子节点？** 每次访问都覆盖从根开始的整条路径，所以父节点的时间 ≥ 它所有后代的时间。只淘汰叶子的 LRU 是自洽的。

**实现要点。** 用最小堆按 `(last_access, tie)` 存候选叶子。弹出一个后检查父节点是否变成了可淘汰的叶子，是的话入堆。复杂度 $O(K\log K)$，$K$ 是候选数。

**和引擎的对应关系。**

- SGLang 的 RadixAttention：树节点存 token 段和对应的 KV 索引。请求开始时 `match_prefix`，然后对命中路径 `inc_lock_ref`；结束时把新生成的部分 `insert` 进树、`dec_lock_ref`。显存不够时 `evict`。
- vLLM 的 Automatic Prefix Caching：不建树，用“前缀哈希 + 块内 token”给每个满块算哈希，用哈希表查找。引用计数为 0 的块按 LRU 进入可淘汰队列。

**易错点。**

- 被锁节点的祖先也都有 ref > 0（锁的是整条路径），所以不会出现“孩子被锁、父亲被删”的情况。
- `match_prefix` 也要刷新访问时间，否则热点前缀会被误删。
- 真实系统里命中长度要按块对齐（vLLM 只缓存满块），KV 布局也要和 attention kernel 匹配。

> [!Q]
> 多轮对话、GRPO 的 n 个采样、few-shot 模板，哪个从前缀缓存里收益最大？
>
> 三者都共享长前缀。GRPO 一个 prompt 采 n 条，prompt 的 prefill 只需要算一次。多轮 agent RL 里，每一轮的上下文都是上一轮的超集，命中率极高。这也是 verl 的 agent loop 要求同一 session 粘在同一个推理实例上（sticky session）的原因。

> [!Q]
> 多个推理实例时，前缀缓存怎么跨实例利用？
>
> 做 cache-aware 路由：按前缀哈希或者近似的 radix 匹配，把请求发给最可能命中的实例（SGLang router 等）。更进一步可以把 KV 放到分布式 KV 存储（例如 Mooncake），让各实例共享。
