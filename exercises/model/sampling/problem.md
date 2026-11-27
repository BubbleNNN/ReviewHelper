---
title: 采样：temperature / top-k / top-p
difficulty: 易
tags: 推理, 采样
order: 22
source: Holtzman et al. 2019 nucleus sampling；HF generate 的 logits warper 顺序
---

对单个位置的 logits（形状 `(V,)` 的 `torch.Tensor`，测试用 float64）实现采样前的处理和采样本身。处理顺序固定为：**temperature → softmax → top-k → top-p**，每一步之后都重新归一化。

```python
top_k_filter(probs, k)            # 返回过滤并归一化后的概率 (V,)
top_p_filter(probs, p)            # 同上
process_logits(logits, temperature=1.0, top_k=0, top_p=1.0) -> probs (V,)
sample(logits, generator, temperature=1.0, top_k=0, top_p=1.0) -> int   # generator: torch.Generator
```

规则（测试按这些规则写）：

- `top_k <= 0` 或 `top_k >= V` 表示不过滤。保留概率 **大于等于第 k 大概率** 的所有 token，也就是边界上并列的全部保留。
- `top_p >= 1.0` 表示不过滤。按概率从大到小排序（并列时下标小的在前，即稳定排序），保留**累计概率第一次 ≥ p 时的最短前缀**，至少保留 1 个。
- `temperature < 1e-5` 视为 greedy：返回在 argmax 处为 1 的 one-hot（并列取下标最小的）。
- `sample` 必须只用传入的 `generator` 产生随机数（例如 `torch.multinomial(probs, 1, generator=generator)`），返回 python `int`。

<!-- solution -->

### 讲解

**temperature**：$p_i \propto \exp(z_i / T)$。$T<1$ 让分布更尖，$T>1$ 更平；$T\to 0$ 退化成 argmax。RL rollout 一般用 $T=1$，这样采样分布就是策略本身，logprob 才能直接拿来算重要性比率。

**top-k**：只在概率最高的 $k$ 个 token 里采样。缺点是 $k$ 固定：分布很尖时 $k$ 个里混进很多垃圾，分布很平时又砍掉了合理选项。

**top-p（nucleus）**：保留累计概率刚好达到 $p$ 的最小集合，集合大小随分布自适应。实现就是“降序排序 → cumsum → 找第一个 ≥ p 的位置”。排序要用 `torch.sort(..., stable=True)`，否则并列 token 的顺序不确定。

**复杂度**：排序 $O(V\log V)$。工程上用 `topk` 先截断（如先取前 1000）再做 top-p，或者用基于阈值的二分避免全排序。

**易错点**

- 先 temperature 再 softmax，不是对概率开 $1/T$ 次方之后忘了归一化。
- top-p 的边界：要保留“让累计概率越过 p 的那个 token”，用 `cumsum - p_i < p` 或 `torch.searchsorted(cumsum, p) + 1` 都可以，注意别差一个。
- top-k 的阈值用 `torch.topk(probs, k).values[-1]`，再 `probs >= kth`，这样边界上并列的 token 都留下。
- 顺序会影响结果：先 top-k 后 top-p 时，top-p 的累计是在 top-k 归一化之后算的。

> [!Q]
> RL 训练时 rollout 用了 top-p=0.9，会有什么问题？
> 实际采样分布是截断后的分布 $\tilde\pi$，而训练时用的 logprob 是在完整分布 $\pi$ 上算的，两者不一致，等于引入了隐蔽的 off-policy 偏差。所以 RL rollout 通常设 `top_p=1, top_k=-1, temperature=1`；如果一定要截断，就需要用推理引擎返回的、与截断分布对应的 logprob 做修正。

```python
# top-p 的另一种常见写法（vLLM / HF 里都是这个思路）：批量处理 (B, V)
sorted_p, idx = probs.sort(descending=True)
cum = sorted_p.cumsum(-1)
remove = cum - sorted_p >= p          # 前面已经够 p 的 token 全部去掉
sorted_p[remove] = 0
probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
probs /= probs.sum()
```
