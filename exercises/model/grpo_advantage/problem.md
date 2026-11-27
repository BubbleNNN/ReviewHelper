---
title: GRPO / Dr.GRPO / RLOO 组内优势
difficulty: 易
tags: RL, GRPO, verl
order: 28
source: verl/trainer/ppo/core_algos.py · compute_grpo_outcome_advantage / compute_rloo_outcome_advantage
---

同一个 prompt 采样 `n` 条回答（`rollout.n`），它们共享同一个 `index`（verl 里是每个 prompt 的 uid）。实现两种只用 outcome reward 的优势估计，对齐 verl。输入是 `torch.Tensor`（测试用 float64），`response_mask` 是 bool：

```python
grpo_advantage(token_level_rewards, response_mask, index,
               norm_adv_by_std=True, epsilon=1e-6) -> (B, T)
rloo_advantage(token_level_rewards, response_mask, index) -> (B, T)
```

- `token_level_rewards`: `(B, T)` float tensor，每条回答的分数 `score = token_level_rewards.sum(-1)`（outcome reward 一般只放在最后一个有效 token 上）。
- `index`: 长度 `B` 的 python list，元素可以是任意可哈希值（字符串 uid、整数）；也可能是 `(B,)` 的 long tensor。同组的回答**不一定相邻**。
- 返回值是把每条回答的标量优势广播到 token 维再乘 `response_mask`，形状 `(B, T)`，dtype 与输入相同，**不带梯度**（和 GAE 一样，优势是常数，用 `torch.no_grad()`）。

**GRPO**：组内 $A_i = \dfrac{s_i - \text{mean}(s)}{\text{std}(s) + \epsilon}$，其中 std 是**无偏估计**（除以 $n-1$，和 `torch.std` 默认一致）。`norm_adv_by_std=False` 时（Dr.GRPO）只减均值不除 std。组里只有 1 条回答时按 mean=0、std=1 处理。

**RLOO**：用“其余 $n-1$ 条的均值”做 baseline：$A_i = s_i - \frac{1}{n-1}\sum_{j\ne i}s_j$。组里只有 1 条时 baseline 为 0（$A_i=s_i$）。

<!-- solution -->

### 讲解

**为什么用组内 baseline**：同一个 prompt 的多条回答，难度相同，组均值就是这个 prompt 下价值函数 $V(s_0)$ 的一个蒙特卡洛估计。这样不需要 critic，省掉一个和 actor 一样大的模型。

**RLOO 和 GRPO（不除 std）的关系**：

$$s_i - \frac{1}{n-1}\sum_{j\ne i}s_j = \frac{n}{n-1}\left(s_i - \bar s\right)$$

只差一个常数缩放。区别在于 leave-one-out baseline 不含 $s_i$ 自己，所以对策略梯度是**严格无偏**的；GRPO 的组均值里含 $s_i$，有 $O(1/n)$ 的偏差（被常数缩放吸收）。

**std 归一化的问题**（Dr.GRPO 的论点）：

- 组内全对或全错时 std=0，优势全为 0，这组样本没有学习信号（DAPO 的 dynamic sampling 会直接过滤掉这种组）。
- std 很小的组（几乎全对、偶尔一个错）会被放大很多，等于给“太简单”或“太难”的题更高权重，引入难度偏差。

**复杂度**：两遍扫描，$O(B)$。verl 里有 `grpo_vectorized` 用 scatter 做向量化。

**易错点**

- `torch.std` 默认无偏（`correction=1`，除以 $n-1$），和 verl 一致；手写 `((s - s.mean())**2).mean().sqrt()` 得到的是有偏版本，对不上。
- 组里只有 1 条时 `torch.std` 返回 nan（$n-1=0$），所以要特判。
- 拿 tensor 当 dict 的 key 分组是坑：tensor 按对象 id 哈希，`{t[0]: ..., t[1]: ...}` 里值相同的元素也会被分到不同组。先 `index.tolist()`。

> [!Q]
> rollout.n 取多大合适？
> n 越大，组均值作为 baseline 越准、方差越小，同时更容易在难题上采到至少一条正确答案（否则全错组没有信号）；代价是 rollout 成本线性增长。数学推理任务常见 n=8 或 16；batch 的 prompt 数要相应调小以控制总 token 数。

```python
# verl 的写法是逐组循环；向量化可以这样（index 先映射成整数 tensor）
uniq, inv = torch.unique(index_tensor, return_inverse=True)
mean = scores.new_zeros(len(uniq)).scatter_reduce(0, inv, scores, "mean", include_self=False)
```
