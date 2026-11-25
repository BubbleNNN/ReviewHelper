---
title: 投机解码：无损性证明、接受长度与加速比
summary: 小模型先猜、大模型一次验证；证明拒绝采样为什么不改变输出分布，推导期望接受长度和加速比，并比较主流草稿方案。
tags: 投机解码, 推理, 概率
order: 70
---

## 为什么能加速

decode 是访存受限的（见“GPU 与 Roofline”）：目标模型一次前向处理 1 个 token 和处理 $\gamma+1$ 个 token，读的权重量几乎一样，耗时也接近。于是：

<div class="flow"><span class="f-accent">草稿模型 q<small>连续猜 γ 个 token</small></span><i>→</i><span class="f-infra">目标模型 p<small>一次前向，并行打分 γ+1 个位置</small></span><i>→</i><span>从左到右逐个接受 / 拒绝</span><i>→</i><span class="f-rl">本轮产出 1 到 γ+1 个 token</span></div>

只要草稿便宜、猜得够准，平均每次目标模型前向就能产出不止 1 个 token。

## 单个 token 的验证规则

设目标分布为 $p(\cdot)$，草稿分布为 $q(\cdot)$（都是给定前缀下的词表分布）。

1. 从草稿采样 $x \sim q$；
2. 以概率 $\min\!\left(1, \dfrac{p(x)}{q(x)}\right)$ **接受** $x$；
3. 若拒绝，从修正分布 $p'(x) = \dfrac{\max(0,\,p(x)-q(x))}{\sum_{x'}\max(0,\,p(x')-q(x'))}$ 重新采样一个 token。

### 证明：输出分布恰好是目标分布

先算总接受概率：

$$\beta = \sum_x q(x)\min\!\left(1,\frac{p(x)}{q(x)}\right) = \sum_x \min\big(p(x),\,q(x)\big)$$

拒绝概率为

$$1-\beta = \sum_x \big(p(x) - \min(p(x),q(x))\big) = \sum_x \max\big(0,\,p(x)-q(x)\big)$$

它恰好等于 $p'$ 的归一化常数。于是对任意 token $x$，输出 $x$ 有两条路径：草稿给出 $x$ 并被接受，或者被拒后从 $p'$ 重采到 $x$：

$$P(\text{out}=x) = q(x)\min\!\left(1,\frac{p(x)}{q(x)}\right) + (1-\beta)\,p'(x) = \min(p(x),q(x)) + \max(0,\,p(x)-q(x)) = p(x)$$

最后一步对 $p(x) \ge q(x)$ 和 $p(x) < q(x)$ 两种情况分别验证即可。**证毕：投机采样是无损的**，输出分布和直接从目标模型采样完全相同，不是近似。

> [!NOTE]
> 贪心解码（temperature = 0）是特例：$p$ 和 $q$ 都退化成 one-hot，规则变成“草稿 token 等于目标 argmax 就接受，否则换成目标 argmax”。

### 一串 token 怎么验

草稿给出 $x_1,\dots,x_\gamma$，目标模型一次前向得到 $p_1,\dots,p_{\gamma+1}$（第 $i$ 个分布以 $x_{<i}$ 为前缀）。从 $i=1$ 开始依次验证：

- 接受 $x_i$ 就继续看 $x_{i+1}$；
- 第一次拒绝时，从 $p'_i$ 重采样一个 token，**丢弃后面所有草稿**（它们的前缀已经不成立）；
- 如果 $\gamma$ 个全被接受，再从 $p_{\gamma+1}$ 免费采一个 **bonus token**。

所以每轮至少产出 1 个 token，最多 $\gamma+1$ 个。被拒绝位置之后的 KV cache 要回滚。

动手：`exercises/system/speculative_sampling` 实现验证函数，并用精确计算检验输出分布等于 $p$。

## 期望接受长度

假设每个位置的接受事件独立、接受率都是 $\alpha$（$\alpha = E[\beta]$）。设被接受的草稿数为 $k$：$P(k)=\alpha^k(1-\alpha)$（$k<\gamma$），$P(\gamma)=\alpha^\gamma$。每轮产出 $k+1$ 个 token，期望为

$$E[\tau] = \sum_{k=0}^{\gamma}\alpha^k = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$$

例：$\alpha=0.8$，$\gamma=4$ 时 $E[\tau]=(1-0.8^5)/0.2 \approx 3.36$。$\gamma \to \infty$ 时上限为 $1/(1-\alpha)=5$。**$\gamma$ 越大收益递减越快**，因为后面的位置要求前面全部被接受。

> [!WARN]
> 独立同分布只是近似。实际中越靠后的位置接受率往往越低（草稿误差会累积），而有的方法后面的位置反而更稳。DSpark 论文专门画了“按位置的条件接受率”曲线，用来说明并行草稿的后缀衰减（suffix decay）问题。

## 加速比

设草稿模型生成一个 token 的耗时是目标模型一次前向的 $c$ 倍，并假设验证 $\gamma+1$ 个 token 的耗时等于一次普通 decode（访存受限的假设）。一轮耗时 $\gamma c + 1$，产出 $E[\tau]$ 个 token（Leviathan et al. 2023）：

$$\text{Speedup} = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$$

例：$\alpha=0.8$，$\gamma=4$，$c=0.05$：$3.36 / 1.2 \approx 2.8$ 倍。$\gamma$ 太大时分母线性增长、分子饱和，所以存在最优 $\gamma$。

> [!Q]
> **为什么高并发时投机解码可能变慢？**
> 上面的公式依赖“验证 $\gamma+1$ 个 token 几乎免费”。batch 很大时目标模型已经接近**算力受限**，多验证一个 token 就多一份计算，还挤占了别的请求的 batch 容量；被拒绝的 token 等于白算。再加上草稿模型本身的开销、可变的接受长度让 CUDA Graph 和 batching 更难做，吞吐反而可能下降。所以生产系统需要根据负载动态调整验证长度，甚至在高负载时关掉投机。

## 主流草稿方案

| 方法 | 草稿怎么来 | 草稿是否自回归 | 验证结构 | 要点 |
|---|---|---|---|---|
| 独立小模型 | 同 tokenizer 的小模型 | 是 | 链 | 最早的方案；小模型要单独部署 |
| Medusa | 目标模型最后一层隐状态上接 K 个额外头，第 k 个头直接预测第 k 个后续 token | 否（各头独立） | 树 | 后面的头不知道前面猜了什么，准确率随位置下降快 |
| EAGLE | 一层 Transformer 在**特征层**自回归：输入目标模型倒数第二层特征 + 下一个 token 的 embedding | 是 | 树（EAGLE-2 按置信度动态建树） | 特征比 token 更好预测 |
| EAGLE-3 | 去掉特征回归损失、直接预测 token；融合目标模型低/中/高多层特征；训练时模拟多步生成（training-time test） | 是 | 树 | 接受率随训练数据量继续提升 |
| MTP（DeepSeek-V3） | 预训练时就带的多 token 预测模块，按顺序串联、共享 embedding 和输出头 | 是（模块间串行） | 链 | 报告第 2 个 token 接受率 85%–90%，生成速度约 1.8 倍（待核实） |
| n-gram / prompt lookup | 从 prompt 或历史里查找匹配的 n-gram | 否 | 链或树 | 无需额外模型，对代码编辑、RAG 很有效 |
| DFlash 类并行草稿 | 一次前向输出整块 $\gamma$ 个 token，目标模型多层隐状态注入草稿每层的 K/V | 否（块内并行） | 链 | 草稿耗时几乎与 $\gamma$ 无关，但后面的位置看不到前面实际采到的 token |

两类草稿的根本取舍：

- **自回归草稿**：后一个 token 以前一个 token 为条件，准确；但草稿耗时随 $\gamma$ 线性增长，只能用很浅的网络、较短的 $\gamma$。
- **并行草稿**：一次出整块，便宜；但每个位置只能预测“边缘分布”，多种可能的续写会混在一起（比如 “of course” 和 “no problem” 混成 “of problem”）。

## 树验证

每个位置不只猜一个 token，而是保留 top-k 个候选，组成一棵 token 树。把树展开成一个序列，用 **tree attention mask**（每个节点只能看到自己的祖先）在一次前向里验证所有分支，再选出被接受的最长路径（多候选也有对应的无损拒绝采样规则，如 SpecInfer）。

- 好处：接受长度明显变长；
- 代价：一次要验证的 token 数是树的大小，比链多得多。低负载时没问题，高负载时这些额外 token 都是实打实的计算。

> [!Q]
> **投机解码在 RL 的 rollout 里好用吗？**
> 有收益也有麻烦。RL 每一步都更新策略权重，草稿模型要跟着同步或重新对齐，否则接受率会随训练逐步下降；rollout 的 batch 通常很大，正处在投机解码收益最小的区间；但长尾阶段（大部分请求已结束、只剩少数超长回复）batch 很小，正好适合投机加速。

## 引向 DSpark

把上面几点合起来，就是 DSpark 要解决的两个问题：**草稿怎样既便宜又不衰减**（并行骨干 + 轻量顺序修正），以及**每个请求该验证多少个 token**（用校准过的置信度估计每个前缀的存活概率，再结合引擎当前负载决定验证预算）。细节见“框架 / DSpark”一章，配套练习 `exercises/system/dspark_scheduler`。
