---
title: MoE 路由：top-k、容量与负载均衡损失
difficulty: 中
tags: MoE, 路由, 专家并行
order: 38
source: Shazeer et al. 2017；Lepikhin et al. 2020 GShard；Fedus et al. 2021 Switch Transformer
---

实现一个 token-choice 的 MoE 路由器。`logits` 形状 `(T, E)`：`T` 个 token、`E` 个专家。

```python
router_probs(logits) -> (T, E)                     # 对专家维做 softmax
top_k_routing(logits, k) -> (expert_idx, weights)  # 各 (T, k)
expert_capacity(num_tokens, num_experts, k, capacity_factor) -> int
dispatch(expert_idx, num_experts, capacity) -> (assignments, dropped)
load_balancing_loss(logits, expert_idx, num_experts) -> float
moe_forward(x, logits, experts, k, capacity) -> y  # x: (T, d)，experts: E 个函数 f(x_rows) -> 同形状
```

约定（测试按这些写）：

- **top-k**：先对**所有专家** softmax，再取概率最大的 k 个，按概率降序排列（并列时下标小的在前），然后把这 k 个权重**重新归一化**到和为 1。
- **容量**：`capacity = ceil(capacity_factor * T * k / E)`。
- **dispatch 优先级**：先按 token 顺序处理所有 token 的**第 1 选择**，再按 token 顺序处理第 2 选择，依此类推。专家已满时这个 (token, slot) 被丢弃。`assignments[e]` 是分配给专家 `e` 的 token 下标列表（按加入顺序），`dropped` 是 `(T, k)` 的 bool 数组。
- **负载均衡损失**：$\mathcal L_{aux} = E\sum_{i} f_i P_i$，其中 $f_i$ = `expert_idx` 中等于 $i$ 的个数 / $(T\cdot k)$，$P_i$ = 所有 token 上专家 $i$ 的 router 概率均值。
- **moe_forward**：$y_t=\sum_{j<k,\ \text{未丢弃}} w_{t,j}\,\text{expert}_{e_{t,j}}(x_t)$，被丢弃的 slot 贡献 0（实际模型里靠残差连接兜底）。

<!-- solution -->

### 讲解

**为什么需要负载均衡损失**：路由是“富者愈富”的：某个专家早期稍好一点就会收到更多 token、被训练得更好、再收到更多 token，最后少数专家过载、多数专家闲置（routing collapse）。$f_i$ 不可导（argmax），$P_i$ 可导；最小化 $\sum f_iP_i$ 会把概率从被过度使用的专家上推开。只要 $f$ 完全均匀（$f_i=1/E$），损失就恰好等于 1，可以拿 1 当参照值看路由是否失衡。

**容量与丢 token**：专家并行时每个专家的输入 buffer 是静态形状（便于 all-to-all 和 kernel），所以要设上限。capacity_factor > 1 留出余量；超出的 token 被丢弃，只走残差。推理时一般不丢（dropless，如 MegaBlocks 用块稀疏矩阵乘实现）。

**和专家并行（EP）的关系**：dispatch 对应一次 all-to-all（把 token 发到专家所在的卡），专家计算后再一次 all-to-all（combine）把结果送回。通信量正比于 $T\cdot k\cdot d$，这是 MoE 训练和推理的主要开销之一。

**其他路由变体**：Mixtral 先在 logits 上取 top-k 再对这 k 个做 softmax（数学上和“全 softmax 后取 top-k 再归一化”等价）；DeepSeek-V3 用 sigmoid 打分加上可调 bias 做 auxiliary-loss-free 的负载均衡；expert-choice 路由让专家挑 token，天然均衡。

> [!Q]
> top-k 后要不要重新归一化权重？
> 两种都有。Switch（k=1）直接用原始概率当门控值，保留“router 有多确定”的信息；Mixtral 对 top-k 重新归一化，输出尺度更稳定。面试时说清楚自己采用哪种，并指出它会影响梯度如何流回 router。

```python
# torch：top-k 路由
probs = logits.softmax(-1)
w, idx = probs.topk(k, dim=-1)
w = w / w.sum(-1, keepdim=True)
```
