---
title: DSpark 是什么：投机解码的两个瓶颈
summary: DeepSeek 在 V4 上线的投机解码框架。先搞清它要解决的两个问题：草稿质量随位置衰减，以及固定长度验证在高负载下浪费算力。
tags: DSpark, 投机解码, DeepSeek, 推理
order: 10
---

> [!NOTE]
> 本章依据 DSpark 论文 *Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*（DeepSeek-AI 与北京大学，arXiv 2607.05147，2026-07）整理。论文后半部分（生产部署、附录）的细节我们只读到摘要级别，文中标“待核实”的地方请以原文为准。二手文章里的数字（如“51%–400%”“2–4×”）和论文口径不一致，面试时引用论文自己的数字。

## 一句话定位

DSpark 是 DeepSeek 开源的**投机解码（speculative decoding）框架**，和 DeepSeek-V4-Flash / V4-Pro 一起部署在生产服务里；配套的草稿模型训练与评测框架叫 **DeepSpec**（同时支持 Eagle3、DFlash、DSpark）。它不改目标模型权重，是纯推理侧的加速。

它有三个组件：

<div class="flow"><span class="f-infra">半自回归草稿模型<small>并行骨干 + 轻量顺序头</small></span><i>→</i><span class="f-accent">置信度头<small>预测每个草稿 token 被接受的概率，并做校准</small></span><i>→</i><span class="f-rl">硬件感知前缀调度器<small>按引擎负载决定每个请求验证多长</small></span></div>

## 从延迟公式看投机解码

一轮“起草 + 验证”的延迟除以这一轮平均接受的 token 数 $\tau$，就是每 token 的延迟：

$$
L=\frac{T_{\text{draft}}+T_{\text{verify}}}{\tau}
$$

所以只有三条路：让起草更便宜、让 $\tau$ 更大、让验证更便宜。现有方法在这三者之间互相拉扯。

### 自回归草稿：贵在 $T_{\text{draft}}$

EAGLE-3、MTP 这类草稿模型一个 token 一个 token 地生成，$T_{\text{draft}}\propto\gamma$（$\gamma$ 为草稿长度）。为了不让起草吃掉收益，只能用很浅的网络、很短的草稿（论文里 EAGLE-3 用 1 层、训练时展开 7 步）。树验证能把 $\tau$ 拉回来，但验证 token 数会膨胀。

### 并行草稿：便宜，但后面的位置越来越不准

DFlash、PARD 这类方法一次前向同时吐出 $\gamma$ 个位置的分布，$T_{\text{draft}}$ 几乎和 $\gamma$ 无关。代价是**每个位置预测的是“对前面所有可能前缀取平均”的边缘分布**，没有以前面实际采样出来的 token 为条件。

论文给了一个很直观的例子：回答可以是 “of course” 也可以是 “no problem”，并行草稿在第 1 位可能采到 “of”，在第 2 位却按边缘分布采到 “problem”，拼出 “of problem”。论文把这叫做 **multi-modal collision**，把越往后接受率越低的现象叫做 **suffix decay**。

> [!KEY]
> 接受是“前缀存活”：第 $k$ 个草稿 token 只有在前 $k-1$ 个都被接受时才有意义。所以期望接受长度是连乘的累加 $\sum_k\prod_{i\le k}\alpha_i$，后面位置的接受率一低，再长的草稿也白搭。

### 第二个瓶颈：验证在高负载下不是免费的

低并发时，目标模型 decode 是访存受限的，多验证几个 token 几乎不增加耗时。但在生产服务里，一个 batch 有成百上千个请求，**每多验证一个 token 都占用 batch 容量**，挤掉别的请求。固定长度验证（每个请求都验 $\gamma$ 个）会把容量浪费在那些大概率会被拒绝的后缀 token 上，而且浪费程度随负载变化。

| | 自回归草稿（EAGLE-3 / MTP） | 并行草稿（DFlash） | DSpark |
|---|---|---|---|
| 起草代价 | $\propto\gamma$ | 约等于常数 | 约等于常数（多一个很小的顺序头） |
| 后位接受率 | 高（有条件） | 衰减 | 高（顺序头修正） |
| 验证长度 | 固定或树 | 固定 | 按置信度和负载动态决定 |

## 为什么 Infra 面试会问它

- 它是“算法 × 系统”结合的典型：草稿模型怎么训、置信度怎么校准是算法问题；验证多长取决于 **SPS(B) 这条硬件曲线**，是系统问题。
- 它把“投机解码在高负载下收益消失”这个工业界痛点讲得很清楚，适合用来回答“投机解码什么时候不划算”。
- 它要求调度决策**不能偷看未来**（non-anticipating），否则会破坏无损性。这是很好的追问点。

> [!Q]
> **投机解码为什么能“无损”？DSpark 改了验证规则吗？**
> 标准投机采样对每个草稿 token 以 $\min(1,p/q)$ 接受，拒绝时从 $\mathrm{norm}(\max(0,p-q))$ 重采样，输出分布严格等于目标分布 $p$。DSpark 用的就是这个链式验证（不是树），在离线评测里温度取 1.0。它额外要求的是：**决定验证多少个 token 的调度器不能依赖未来采样结果**。推导见“大模型 Infra / 投机解码”一篇，动手题见 `exercises/system/speculative_sampling`。

> [!Q]
> **既然高负载下投机解码收益下降，为什么不直接关掉？**
> 关掉等于放弃低负载时的加速；而在高负载时，“少验证一点”往往比“完全不验证”更好，因为前一两个草稿 token 的接受率通常很高，验证它们几乎是净赚。DSpark 的调度器就是在每一步按负载连续地调这个长度，而不是开/关二选一。
