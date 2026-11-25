---
title: 投机采样的验证步（无损拒绝采样）
difficulty: 中
tags: 投机解码, 推理加速, 概率
order: 26
source: Leviathan et al. (ICML'23) · Chen et al. (2023) Speculative Sampling
---

draft 模型已经在 $\gamma$ 个位置上依次采出了 token $x_1..x_\gamma$（第 $k$ 个位置的 draft 分布是 $q_k$）。target 模型一次前向就给出了这 $\gamma+1$ 个位置的分布 $p_1..p_{\gamma+1}$，最后一个位置用于 bonus token。请实现**链式验证**：

1. 从 $k=1$ 开始，以概率 $\min\!\left(1, \frac{p_k(x_k)}{q_k(x_k)}\right)$ 接受 $x_k$。具体判定：`u_accept[k] < p[k, x] / q[k, x]` 时接受。
2. 第一次拒绝发生在位置 $k$ 时，从残差分布 $\mathrm{norm}\big(\max(0,\ p_k - q_k)\big)$ 采一个**修正 token**，然后停止。
3. 如果 $\gamma$ 个全部接受，再从 $p_{\gamma+1}$ 采一个 **bonus token**。

采样统一用模板里给好的逆 CDF 函数 `sample_from(probs, u_sample)`（返回最小的 $t$ 使得累积概率 $> u$），保证结果确定、可测试。

```python
verify(p, q, draft_tokens, u_accept, u_sample) -> list[int]
# p: (γ+1, V)   q: (γ, V)   draft_tokens: 长度 γ
# 返回 接受的 draft token + 1 个修正/bonus token，长度在 1..γ+1
```

代码里下标从 0 开始：`p[k]`、`q[k]`、`draft_tokens[k]`、`u_accept[k]` 对应第 $k+1$ 个位置。

<!-- solution -->

### 讲解

**为什么输出分布严格等于 $p$？** 只看一个位置。设 $\alpha = \sum_v \min(p(v), q(v))$。输出 $v$ 有两条路：

- draft 采到 $v$ 并被接受：$q(v)\cdot\min\!\left(1,\frac{p(v)}{q(v)}\right) = \min(p(v), q(v))$
- draft 采到任意 token 被拒绝（总概率 $1-\alpha$），再从残差采到 $v$：$(1-\alpha)\cdot\frac{\max(0,\ p(v)-q(v))}{\sum_w \max(0,\ p(w)-q(w))}$

又因为 $\sum_w \max(0, p(w)-q(w)) = 1-\alpha$，第二项就是 $\max(0, p(v)-q(v))$。两项相加：

$$\min(p,q) + \max(0, p-q) = p$$

对每个位置归纳，整条序列的分布和直接从 target 采样完全一样，所以是**无损**的。测试里的“精确分布检查”用均匀网格对随机数做数值积分，验证的就是这个式子。

**期望接受长度。** 位置 $i$ 的接受率是 $\alpha_i$（各位置独立时），期望接受的 draft 数是

$$\mathbb{E}[\text{accepted}] = \sum_{k=1}^{\gamma}\prod_{i=1}^{k}\alpha_i$$

每轮还会多产出 1 个修正或 bonus token。所有 $\alpha_i=\alpha$ 时，每轮期望产出 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 个 token。

**易错点。**

- 拒绝后要**立刻停止**，后面的 draft token 全部作废（它们的条件前缀已经变了）。
- 残差要先截断负数再归一化，而不是 $p-q$ 直接归一化。
- 贪心解码（temperature=0）时退化为“draft token 等于 target argmax 就接受”。
- 全部接受时 bonus token 是白赚的：target 在最后一个位置的分布本来就算出来了。

> [!Q]
> 为什么高并发时投机解码收益变小甚至变负？
>
> 投机解码用多出来的计算（验证 $\gamma+1$ 个 token）换更少的 decode 步数。低负载时 decode 是访存受限，多算几个 token 几乎免费。高负载时 batch 已经把算力吃满，每个被拒绝的 token 都是纯浪费。DSpark 的做法是根据置信度和引擎负载，动态决定每个请求验证多长（见 exercises/system/dspark_scheduler）。

> [!Q]
> 树形验证（tree verification）比链式好在哪？
>
> 每个位置保留多个候选，组成一棵树，用 tree attention mask 一次前向验证所有分支，取最长的被接受路径。期望接受长度更高，代价是验证的 token 数变多。DSpark 论文用的是链式验证，后续的 AdaSpark 引入了树验证。
