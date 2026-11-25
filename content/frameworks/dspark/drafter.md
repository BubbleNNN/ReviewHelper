---
title: 半自回归草稿模型与 DeepSpec 训练
summary: 一个重的并行骨干负责“看上下文”，一个极轻的顺序头负责“对齐前面采样到的 token”；再加三个损失一起训练。
tags: DSpark, 草稿模型, 训练
order: 20
---

## 结构总览

<div class="flow"><span>目标模型隐状态<small>若干层拼接</small></span><i>→</i><span class="f-infra">KV 注入<small>RMSNorm(W_c[·]) 拼进每层 K/V</small></span><i>→</i><span class="f-infra">并行骨干（DFlash 式）<small>一次前向得到 γ 个位置的 logits U_k</small></span><i>→</i><span class="f-accent">顺序头<small>U_k + B_k(前一个采样 token)</small></span><i>→</i><span class="f-rl">左到右采样草稿</span></div>

### 并行阶段

- 骨干沿用 DFlash，并做了一处修改：**anchor**（目标模型刚产出的最后一个 token）本身作为第一个预测位置。输入是 anchor + $\gamma-1$ 个 mask token，一次前向得到 $\gamma$ 个位置的草稿 logits $U_1,\dots,U_\gamma$。
- 目标模型的上下文通过 **KV injection** 注入：取目标模型若干层的隐状态拼接，经 $\mathrm{RMSNorm}(W_c[\cdots])$ 投影，再拼进草稿模型**每一层**的 key/value。草稿模型因此不用自己从头编码整段上下文。
- 草稿模型和目标模型**共享 embedding 和 LM head**，二者都冻结。

### 顺序阶段

在每个位置的 logits 上加一个只依赖“前面已采样 token”的偏置，得到真正的因果分解：

$$
p_k(v\mid x_0,x_{<k})=\mathrm{softmax}\big(U_k(v)+B_k(x_0,x_{<k},v)\big)
$$

两种顺序头：

| | Markov 头（默认） | RNN 头 |
|---|---|---|
| 条件 | 只看上一个采样 token $x_{k-1}$ | 门控隐状态累积整个前缀 |
| 形式 | $B(x_{k-1},\cdot)=W_1[x_{k-1}]\,W_2$，低秩 $r=256$（$W_1\in\mathbb R^{V\times r}$ 查表，$W_2\in\mathbb R^{r\times V}$） | $s_k=\sigma(W_g z_k)\odot s_{k-1}+(1-\sigma(W_g z_k))\odot\tanh(W_c z_k)$，$B_k=W_2^\top\tanh(W_o z_k)$ |
| 效果 | 已经能解决大部分 multi-modal collision | 只在长草稿时略有提升，部署更复杂 |

> [!KEY]
> 设计思想：把贵的部分（理解上下文、给出每个位置的大致分布）并行做一次；把便宜但必须顺序做的部分（和前一个采样 token 对齐）压成一个低秩查表。所以起草代价几乎不随 $\gamma$ 增长：论文报告 batch=128 时，草稿长度从 4 增到 16，单轮延迟只比 DFlash 多 0.2%–1.3%。

> [!Q]
> **Markov 头只看前一个 token，为什么够用？**
> 并行骨干已经给了每个位置以完整上下文为条件的分布，缺的只是“和刚采样出的那个 token 保持一致”。多模态冲突大多发生在相邻 token 之间（“of course” / “no problem”），一阶条件就能把它们分开。论文的消融也显示 RNN 头的额外收益很小。

## 规模

| 设置 | 草稿结构 |
|---|---|
| 离线实验（Qwen3-4B/8B/14B、Gemma4-12B） | DSpark 与 DFlash 都用 5 层；Eagle3 用 1 层（训练展开 7 步）；block size 7 |
| 生产（DeepSeek-V4，称为 DSpark-5） | 最大 $\gamma=5$；骨干是 3 个 MoE 层，带 mHC 和窗口 128 的滑动窗口注意力；Markov 头 |

> [!WARN]
> 论文同时出现 “block size 7” 和 “$\gamma$ 个草稿 + 1 个 anchor” 的说法，离线实验里 $\gamma$ 到底是 6 还是 7 不完全清楚（待核实）。另一个反直觉的结论：**2 层的 DSpark 在所有领域都超过 5 层的 DFlash**，说明瓶颈在条件结构，不在深度。

## 训练：DeepSpec

目标模型冻结；训练骨干、顺序头和置信度头（置信度头见下一篇），embedding 和 LM head 冻结。每条目标序列上采样多个 anchor，构成 $\gamma$ 长的训练块。

三个损失都按位置加权 $w_k=e^{-(k-1)/\gamma}$（越靠前越重要，因为前缀存活）：

$$
\mathcal L_{\text{ce}}=-\sum_k w_k\log p^{d}_k(x^*_k),\qquad
\mathcal L_{\text{tv}}=\sum_k w_k\,\big\lVert p^{d}_k-p^{t}_k\big\rVert_1
$$

$$
\mathcal L=0.1\,\mathcal L_{\text{ce}}+0.9\,\mathcal L_{\text{tv}}+1.0\,\mathcal L_{\text{conf}}
$$

- $\mathcal L_{\text{ce}}$：对目标模型实际生成的 token 做交叉熵；
- $\mathcal L_{\text{tv}}$：草稿分布和目标分布的 $L_1$ 距离，是**接受率的直接代理**：投机采样的单步接受率恰好是 $\sum_v\min(p^d,p^t)=1-\tfrac12\lVert p^d-p^t\rVert_1$；
- $\mathcal L_{\text{conf}}$：置信度头的 BCE。

数据：Open-PerfectBlend 约 130 万 prompt（对话 17.6%、数学 39.4%、代码 38.9%、指令遵循 4.1%），response 由各自的目标模型在 non-thinking 模式下重新生成（让草稿学的是目标模型自己的分布）；每个草稿模型训练 10 个 epoch。对比基线也在同一框架、同一数据上重训。

生产训练（内部 HAI-LLM 框架）的两个工程优化：

1. **只传 LM head 之前的隐状态**，LM head 在本地对采样位置计算，通信量从 $O(V)$（$V\approx10^5$）降到 $O(d)$；
2. **anchor 边界的序列打包**：用 token 级的注意力索引表示块边界，不构造二维 mask。

> [!Q]
> **为什么训练用 TV 距离作为主损失，而不只用交叉熵？**
> 交叉熵只拟合目标模型采样出的那一个 token，是对目标分布的单样本估计；投机采样关心的是两个**分布**的重合度 $\sum\min(p^d,p^t)$，它正好等于 $1-\mathrm{TV}$。直接优化 TV 就是直接优化接受率。权重 0.9 vs 0.1 也说明了这一点。

> [!Q]
> **为什么 response 要用目标模型重新生成，而不是用数据集原有答案？**
> 草稿模型要模仿的是目标模型的分布，而不是人类答案的分布。用原答案训练，草稿学到的分布会和目标模型错位，接受率下降。这也是蒸馏类方法的通用做法。

> [!TIP]
> 二手文章提到 DeepSpec 的数据准备要缓存目标模型隐状态，Qwen3-4B 的缓存量约 38 TB，默认配置假设单机 8 卡（来自二手文章，待核实）。面试时可以用来说明“草稿模型训练的瓶颈在 I/O 和存储，而不在算力”。
