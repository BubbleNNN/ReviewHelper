---
title: Trie 与前缀匹配
difficulty: 中
tags: 字典树, 字符串, DFS
order: 32
source: LeetCode 208 扩展 · prefix cache 命中长度
---

实现一棵字典树 `Trie`：

```python
t = Trie()
t.insert("apple")
t.search("apple")          # True：完整插入过的词
t.search("app")            # False
t.starts_with("app")       # True：是某个已插入词的前缀
t.longest_prefix("applet") # 5："appl" "e" 都能在树里走到，"t" 走不到
t.complete("ap")           # ["apple"]：以 "ap" 开头的所有已插入词，字典序
t.complete("ap", limit=0)  # []
```

| 方法 | 说明 |
|---|---|
| `insert(word)` | 插入一个词；重复插入同一个词只算一次 |
| `search(word) -> bool` | word 是否被完整插入过 |
| `starts_with(prefix) -> bool` | 是否存在以 prefix 开头的已插入词；空串返回“树非空” |
| `longest_prefix(word) -> int` | 从根出发，沿着 word 的字符能走多远（word 与“所有已插入词的前缀集合”的最长公共前缀长度） |
| `complete(prefix, limit=None) -> list[str]` | 以 prefix 开头的已插入词，按字典序；`limit` 不为 None 时最多返回 limit 个 |

场景：SGLang 的 RadixAttention 把所有请求的 token 序列存在一棵 radix tree（压缩的 trie）里。新请求来了，先做一次 `longest_prefix`，命中的那段 KV cache 直接复用，只需要 prefill 剩下的部分。

<!-- solution -->

### 讲解

**节点结构**：`children: dict[str, Node]` + `is_end: bool`。

- `insert` / `search` / `starts_with` 都是沿字符往下走，$O(L)$。
- `longest_prefix`：一直往下走到走不动为止，返回走过的字符数。
- `complete`：先走到 prefix 对应的节点，再对子树做 DFS，**按字符升序**访问孩子，得到的就是字典序。`limit` 到了就提前停止。

**复杂度**：插入/查询 $O(L)$；`complete` 是 $O(L + \text{输出总长度})$（加上对孩子排序的开销）。空间最坏是所有词长度之和。

**易错点**：

- `search` 走到了终点还要检查 `is_end`，否则 "app" 会被误判为存在。
- 字典序 DFS 时，当前节点是一个完整词要**先**输出，再递归孩子（"app" 排在 "apple" 前面）。
- 用 dict 存孩子更通用；只有小写字母时可以用长度 26 的数组，更快但更占内存。

> [!Q]
> **Radix tree 和 Trie 有什么区别，为什么 SGLang 用 radix tree？** Radix tree 把只有一个孩子的链压缩成一条边（边上存一段序列），节点数和分叉数成正比而不是和总长度成正比。token 序列很长、分叉不多，压缩后内存和遍历开销都小很多。插入时如果新序列在一条边的中间分叉，要把这条边**分裂**成两段。

> [!Q]
> **prefix cache 满了淘汰谁？** 只能淘汰没有正在运行的请求引用的节点，一般从**叶子**开始按 LRU 淘汰（叶子被删后父节点可能变成新叶子）。内部节点被多个请求共享，最后才淘汰。
