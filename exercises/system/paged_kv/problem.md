---
title: PagedAttention 块管理器（引用计数 + 写时复制）
difficulty: 中
tags: KV cache, vLLM, 内存管理
order: 20
source: vLLM PagedAttention (Kwon et al., SOSP'23) · BlockSpaceManager
---

实现一个 PagedAttention 风格的 KV cache 块管理器。物理显存被切成 `num_blocks` 个块，每块能放 `block_size` 个 token 的 KV。每条序列有一张 **block table**（逻辑块号 → 物理块号）。块编号是 `0..num_blocks-1`，空闲块的分配顺序不限。

```python
bm = BlockManager(num_blocks, block_size)
bm.allocate(seq_id, n_tokens)   # 新序列，放入 n_tokens 个 prompt token，占 ceil(n/bs) 块
bm.append_token(seq_id)         # 生成一个新 token；返回 None 或 (src_block, dst_block)
bm.fork(parent_id, child_id)    # child 共享 parent 的所有块（引用计数 +1），用于 beam search / 并行采样
bm.free(seq_id)                 # 引用计数 -1，降到 0 的块回到空闲池
bm.block_table(seq_id) -> list[int]
bm.num_tokens(seq_id) -> int
bm.num_free_blocks() -> int
bm.ref_count(block_id) -> int   # 空闲块为 0
```

规则：

- **写时复制（copy-on-write）**：`append_token` 时，如果最后一个块还没满、但被多条序列共享（`ref_count > 1`），就分配一个新块，旧块引用计数减 1，本序列 block table 的最后一项换成新块，返回 `(旧块, 新块)` 表示需要把旧块内容拷贝过去。其他情况返回 `None`。
- 最后一个块已满时，直接分配一个新块（不需要拷贝），返回 `None`。
- 空闲块不够时抛 `MemoryError`，并且**不改变任何状态**（`allocate` 不能只分配一半）。
- `allocate` 一个已经存在的 `seq_id`，或者 `fork` 到已存在的 `child_id`，抛 `ValueError`；查询不存在的序列抛 `KeyError`（用 dict 存就是默认行为）。

<!-- solution -->

### 讲解

**数据结构。** 一个空闲块栈（list），`ref[block]` 数组，`tables[seq] = list[block]`，`ntok[seq]`。所有操作都是 O(1) 或 O(块数)。

**append 的三种情况：**

1. `ntok % block_size == 0`：最后一块满了（或序列为空），要新块。
2. 最后一块没满且 `ref == 1`：直接写，什么都不用做。
3. 最后一块没满且 `ref > 1`：写时复制。

**为什么只有最后一块需要 CoW？** 前面的块都是满的、只读的，KV 一旦写入就不会再改。所以 fork 出来的 $n$ 个样本可以一直共享 prompt 的 KV，只在各自分叉的那一块上分开。这正是 GRPO 一个 prompt 采 `rollout.n` 条的场景：prompt 部分的 KV 只存一份。

**碎片。** 每条序列只有最后一块可能没装满，内部碎片 < `block_size` 个 token；物理块可以不连续，没有外部碎片。论文里说现有系统有 60%–80% 的 KV 显存被浪费，vLLM 把浪费降到 4% 以下。

**易错点。**

- CoW 时要先确认有空闲块再改 block table，否则失败后状态是坏的。
- `free` 时只把引用计数降到 0 的块放回空闲池，共享块不能提前释放。
- 块大小是权衡：太小时 block table 变长，gather 访存的效率下降；太大时内部碎片多。vLLM 在 GPU 上默认是 16。

> [!Q]
> prefix caching 和这里的 fork 有什么区别？
>
> fork 是同一个请求内部显式共享（并行采样、beam search）。prefix caching 是**跨请求**复用：用块内 token 的哈希（连同前缀的哈希）做 key，新请求的前缀命中时直接复用物理块，引用计数为 0 的块按 LRU 淘汰。SGLang 用 radix tree 做同样的事。

> [!Q]
> 显存不够时 vLLM 怎么办？
>
> 抢占（preemption）：把某些序列换出到 CPU（swap），或者直接丢掉 KV、稍后重新 prefill（recompute）。V1 引擎默认走 recompute，调度器按优先级 / FCFS 选择牺牲者。
