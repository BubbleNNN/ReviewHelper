---
title: DSpark 硬件感知前缀调度器
difficulty: 难
tags: DSpark, 投机解码, 调度, 贪心
order: 28
source: DSpark arXiv 2607.05147 · Algorithm 1 (Hardware-Aware Prefix Scheduler)
---

DSpark 的 draft 模型对每个请求一次性给出 $\gamma$ 个 draft token，同时给出每个位置的**条件置信度** $c_{r,j}$：在前 $j-1$ 个都被接受的前提下，第 $j$ 个被接受的概率。调度器要决定每个请求实际送去 target 验证的长度 $l_r \in [0, \gamma_r]$。

**符号**

- 前缀存活概率：$a_{r,j} = \prod_{i\le j} c_{r,i}$，即前 $j$ 个 draft token 全部被接受的概率。它随 $j$ 单调不增。
- 验证 batch 的 token 数：$B = \sum_r (1 + l_r)$。每个请求至少要验证 1 个位置（产出修正或 bonus token）。
- 期望产出 token 数：$\tau = \sum_r \big(1 + \sum_{j\le l_r} a_{r,j}\big)$
- 吞吐目标：$\Theta = \tau \cdot \mathrm{SPS}(B)$。$\mathrm{SPS}(B)$ 是引擎在 batch 为 $B$ 时每秒能跑多少 step，由启动时 profile 出的成本表给出。

**Algorithm 1（贪心 + 早停）**

1. 算出所有 $a_{r,j}$，候选集 $E=\{(r,j) : a_{r,j}>0\}$，按 $a_{r,j}$ **降序**排序（同值按 $r$、$j$ 升序）。
2. 初始化 $l_r=0$，$B=R$，$\tau=R$，$\Theta_{best}=R\cdot\mathrm{SPS}(R)$。
3. 依次取 $(r,j)$：令 $l_r=j$，$B \mathrel{+}= 1$，$\tau \mathrel{+}= a_{r,j}$，算 $\Theta$。如果 $\Theta > \Theta_{best}$ 就记下当前方案，否则**立即停止**。

```python
prefix_survival(conf) -> list[list[float]]
greedy_schedule(conf, sps) -> (lengths, theta)       # Algorithm 1
brute_force_schedule(conf, sps) -> (lengths, theta)  # 枚举所有组合，用来对拍
# conf: 长度 R 的 list，conf[r] = [c_{r,1}, ..., c_{r,γ_r}]（可以为空）
# sps:  callable, B -> float
```

<!-- solution -->

### 讲解

**为什么全局排序还能保证“前缀”？** 请求 $r$ 验证到第 $j$ 个位置，前提是前面的位置都要验证（链式验证，前面拒绝了后面就作废）。因为 $a_{r,j}$ 随 $j$ 单调不增，同值时又按 $j$ 升序，所以同一个请求的候选在排序里一定按 $j=1,2,\dots$ 的顺序出现。贪心选中的永远是每个请求的一段前缀。而且把请求 $r$ 从 $j-1$ 扩到 $j$，$\tau$ 的边际收益正好是 $a_{r,j}$。

**为什么贪心 + 早停最优（在条件满足时）？** $\mathrm{SPS}$ 只依赖总 token 数 $B$。固定接纳 $K$ 个位置时，$\tau$ 最大的方案就是取最大的 $K$ 个 $a$，即贪心的前 $K$ 步。于是问题变成一维的：在 $K$ 上最大化 $\Theta(K)$。$\tau(K)$ 是凹的（增量递减）。若 $\mathrm{SPS}(B)=1/(c_0+c_1B)$ 这类形式，$\Theta$ 是单峰的，第一次下降就是全局最优。测试里的 `jagged` 例子展示了论文提到的问题：真实 SPS 有台阶，早停会卡在局部最优。

**负载自适应。** 低负载时 SPS 平坦，多验证几个 token 几乎免费，调度器会把能验证的都验证掉。高负载时 SPS 陡峭，预算自动收缩。这是静态阈值做不到的。论文报告中等负载时每个请求的验证预算约 4–6 个 token，MTP-1 基线是固定的 2 个；负载升高后这个预算会平滑缩小。

**无损性：non-anticipating。** 投机采样无损的前提是：是否验证位置 $k$ 的决定，不能依赖位置 $k$ 本身还没揭晓的采样结果。DSpark 的 $c_{k+1}$ 依赖已经采出的 draft token $x_k$（Markov head 的特征）。如果做全局回溯式搜索，就可能把 $x_k$ 的信息漏进“要不要验证 $k$”的决定里，输出分布就偏了（论文附录给了 $\gamma=2$ 的反例）。早停保证每个决定只依赖已经处理过的前缀。

**生产版本（DeepSeek-V4 上的 ZOS 适配）。** Zero-Overhead Scheduling 要求在当前 step 结束前就确定下一步的 batch 大小。所以生产实现用**两步之前**的置信度估计容量 $K$，再按当前的累积置信度选 top-$K$，同时去掉了早停，以绕开 SPS 的台阶。论文用“因果屏障”论证这仍然无损：截断长度只依赖两步之前的信息。我们的读法是，这个论证论文是断言而没有给出形式化证明。

**复杂度。** 排序 $O(R\gamma\log(R\gamma))$，扫描 $O(R\gamma)$。暴力是 $O(\prod_r(\gamma_r+1))$，只能用来对拍。

> [!Q]
> 置信度为什么需要校准（calibration），只要排序对不就行了吗？
>
> 不行。调度器把 $a_{r,j}$ 当成真实概率去算期望 $\tau$，再和 SPS 的代价比较，绝对数值直接决定停在哪里。DSpark 用 Sequential Temperature Scaling 逐位置做温度缩放，把累积置信度的 ECE 从 3%–8% 降到约 1%。

> [!Q]
> 如果某个请求的 draft 质量特别差，这个调度器会怎么处理它？
>
> 它的 $a_{r,1}$ 很小，在全局排序里排在后面，高负载时基本只验证 1 个位置（$l_r=0$，只出修正或 bonus token），等价于普通 decode。计算预算会自动让给 draft 质量好的请求。但 draft 的前向成本（固定 $\gamma$ 个位置）仍然要付，这是论文列出的局限之一。
