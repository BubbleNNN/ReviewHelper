---
title: LLM-RL 算法与实践
summary: GRPO 家族、DPO、KL 与熵、loss 聚合、训推不一致和训练排障。
tags: RL, LLM
order: 20
---

## 怎么把 LLM 生成建模成 MDP？

状态 $s_t = (x, y_{\lt t})$，动作 $a_t = y_t\in\mathcal V$，转移是确定性拼接，奖励通常只在 EOS 处给 $R(x,y)$（可选逐 token KL 惩罚），episode 在 EOS 或最大长度结束。也可以把整条回答当一个动作，看成 contextual bandit（GRPO / RLOO 的视角）。多轮 agent 中工具返回的 token 不是动作，要 mask 掉。

## RLHF 的三个阶段分别做什么？

1. SFT：示范数据监督微调，得到初始策略和参考模型；
2. 奖励模型：人类对回答两两比较，用 Bradley-Terry 损失 $-\log\sigma(r_w - r_l)$ 训练标量 RM；
3. RL：PPO 最大化 RM 分数，减去与 SFT 模型的 KL，防止 reward hacking 和语言漂移。

## 推导 DPO，为什么配分函数 Z(x) 能消掉？

KL 约束目标的最优解 $\pi^\star \propto \pi_{\text{ref}}\exp(r/\beta)$，反解 $r = \beta\log\frac{\pi^\star}{\pi_{\text{ref}}} + \beta\log Z(x)$。代入 BT 模型只用到同一 $x$ 下两条回答的奖励差，$\beta\log Z(x)$ 相减抵消，得到

$$\mathcal L = -\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\Big)$$

## DPO 训练时 chosen 的 logprob 也下降，为什么？怎么办？

损失只要求 chosen 与 rejected 的隐式奖励差变大，两者可以同时下降，概率质量流向数据外的回答，可能伤害生成质量。办法：在 chosen 上加 NLL 项、调 β、用当前策略生成新的偏好对做迭代 / on-policy DPO。

## GRPO 的优势怎么算？相比 PPO 的优缺点？

同一 prompt 采 G 条，$\hat A_i = (r_i - \text{mean})/(\text{std}+\varepsilon)$，序列内所有 token 共享。KL 用 k3 放进 loss。优点：不需要 critic，省一半模型显存，实现简单。缺点：rollout 成本 ×G；没有 token 级信用分配；全对 / 全错组无信号；std 归一化有难度偏置。verl 中 std 用 `torch.std`（无偏）。

## GRPO 的 std 归一化有什么问题？

- 难度偏置：几乎全对或全错的题 std 很小，少数异类样本的优势被放大，梯度被这些题主导；
- baseline 和缩放都依赖当前样本本身，估计有偏；
- Dr.GRPO 去掉除 std（`norm_adv_by_std_in_grpo=False`），REINFORCE++ 改用全局 batch 标准化。

## Dr.GRPO 改了 GRPO 的哪两处？

1. 去掉优势的 std 归一化（消除难度偏置）；
2. 去掉 $1/|o_i|$ 的序列长度归一化，改成 token 求和再除以常数（消除长度偏置：原写法让长的错误回答每个 token 受罚更轻）。

verl：`loss_agg_mode=seq-mean-token-sum-norm`、`norm_adv_by_std_in_grpo=False`、`use_kl_loss=False`。

## DAPO 做了哪些改动？

- Clip-Higher：$\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$；
- Dynamic Sampling：过滤全对 / 全错的组，不够就继续采样；
- Token-level loss：token-mean 聚合；
- Overlong reward shaping：长度进入缓冲区后线性扣分；
- 去掉 KL。

论文在 Qwen2.5-32B base 上报告 AIME 2024 50 分。

## clip-higher 为什么能缓解熵塌缩？

对称 clip 下 ratio 上限是 $1+\epsilon$：概率 0.01 的 token 一次最多到 0.012，而高概率 token 很容易被推到接近 1。结果低概率的“探索性” token 永远涨不上来，分布越来越尖。放宽上界让低概率、正优势的 token 能被明显提升，增加多样性；下界保持 0.2，避免坏 token 被压得过快。

## RLOO 的 baseline 是什么？为什么无偏？

$\hat A_i = r_i - \frac{1}{k-1}\sum_{j\ne i}r_j$，用组内其他样本的均值做 baseline。它与第 $i$ 个样本独立，所以 $\mathbb E[b\nabla\log\pi(y_i)] = 0$，梯度无偏。它等于 $\frac{k}{k-1}(r_i - \bar r)$，与减组均值只差常数倍。

## REINFORCE++ 与 GRPO 有什么不同？

REINFORCE++ 不做组内标准化，而是对整个 batch 做全局优势标准化；KL 作为 token 级惩罚放进 reward；仍用 PPO-clip，不要 critic。baseline 版先减组均值再全局标准化。动机是组内标准化容易让模型过拟合简单题，全局标准化让尺度更稳定。

## GSPO 为什么要用序列级 ratio？

奖励是序列级的，而 token 级 ratio 每个 token 只有一个样本，噪声大，长序列上累积，MoE 路由变化时尤其不稳。GSPO 用长度归一化的序列比值 $(\pi_\theta(y)/\pi_{\text{old}}(y))^{1/|y|}$ 并在序列级 clip，整条回答一起保留或截断梯度。因为是几何平均，clip 范围要小得多（论文在 $10^{-4}$ 量级）。

## KL 惩罚放 reward 和放 loss 有什么区别？

放 reward：每 token 奖励减 $\beta\log\frac{\pi_{\text{old}}}{\pi_{\text{ref}}}$，经 GAE 进入优势，后续偏离会影响前面 token，KL 值不求导（经典 PPO-RLHF，verl `use_kl_in_reward`）。放 loss：直接把 KL 估计量加到 loss 上逐 token 求导，局部正则（GRPO，verl `use_kl_loss`）。两者优化的目标并不等价。

## k1、k2、k3 三个 KL 估计量分别是什么？

$r = \pi_{\text{ref}}/\pi_\theta$，样本来自 $\pi_\theta$，估计 $\text{KL}(\pi_\theta\|\pi_{\text{ref}})$：

- k1 $= -\log r$：无偏，方差大，可为负；
- k2 $= \tfrac12(\log r)^2$：有偏但偏差小，方差小；
- k3 $= r - 1 - \log r$：无偏、非负、方差小（verl 的 `low_var_kl`）。

## 把 k1 或 k3 当 loss 求导，梯度是什么？

样本固定、只对 $\log\pi_\theta$ 求导：$\nabla k1 = \nabla\log\pi_\theta$，期望为 0，没有正则效果；$\nabla k3 = (1 - r)\nabla\log\pi_\theta$，期望是正向 $\text{KL}(\pi_{\text{ref}}\|\pi_\theta)$ 的梯度；只有 k2 的梯度期望等于反向 KL 的梯度。所以 verl 提供 `k3+`：值用 k3，梯度用 k2（straight-through）。

## 什么是熵塌缩？怎么监控、怎么缓解？

RL 训练初期策略熵快速下降，回答趋同，探索不足导致性能饱和。研究发现熵的变化由 log-prob 与优势的协方差驱动，性能近似 $R = -a e^{\mathcal H} + b$。监控 `actor/entropy`、组内回答多样性、零优势组比例。缓解：clip-higher、Clip-Cov / KL-Cov、谨慎的熵 bonus、过滤过简单的数据、保持 rollout 温度。

## loss_agg_mode 几种模式有什么区别？

- `token-mean`：所有有效 token 平均，每 token 权重 $1/N$（verl 默认，DAPO 用）；
- `seq-mean-token-mean`：先序列内平均再序列间平均，长序列 token 权重被稀释（GRPO 原文，有长度偏置）；
- `seq-mean-token-sum`：序列内求和再平均，长序列权重大；
- `seq-mean-token-sum-norm`：再除以常数，每 token 权重与长度无关（Dr.GRPO）。

## 为什么 verl 的 token-mean 要除以全局 token 数再乘 dp_size？

每个 micro-batch 只算自己 token 的损失和，除以整个 mini-batch 在所有 rank 上的 token 总数；梯度累积把 micro-batch 加起来；DDP/FSDP 的 all-reduce 对 rank 求平均，相当于除以 dp_size，所以先乘回去。最终等于全局 token 平均，与 micro-batch 切分和各 rank 的 token 分布无关。若每个 micro-batch 先各自求平均，token 数不同的 micro-batch 权重就错了。

## 什么是训练-推理不一致？TIS 和 IcePop 怎么修？

推理引擎（vLLM/SGLang）和训练引擎（FSDP/Megatron）对同一 token 算出的 logprob 不同（kernel、精度、MoE 路由），名义上的 on-policy 实际是 off-policy，长期累积会导致崩溃。修正：权重 $\pi_{\text{old}}/\pi_{\text{rollout}}$ 乘到 loss 上；TIS 截断上界（verl `rollout_is_threshold=2.0`）；IcePop 把比值在 $[0.5, 5]$ 之外的 token 权重置 0（阈值写 `"0.5_5.0"`）；还可以用 `rollout_rs` 直接剔除。

## 一个 prompt 的回答全对或全错怎么办？

组内优势全 0，不产生梯度却占算力、稀释 token-mean。办法：DAPO 的 dynamic sampling（verl `algorithm.filter_groups`，不够就继续生成）；离线按 pass rate 过滤数据；增大 `rollout.n`；课程学习。并监控零优势组的比例。

## 回答被最大长度截断，奖励怎么处理？

截断回答多半拿 0 分，正确但冗长的推理被惩罚，形成奖励噪声。选择：overlong filtering（截断样本不计入 loss）；DAPO 软惩罚（超过 $L_{\max}-L_{\text{buf}}$ 后线性扣分，verl `overlong_buffer`）；逐步加长上下文；监控 `response_length/clip_ratio`。

## 举几个 reward hacking 的例子和对策

例子：RM 偏好长回答导致越写越长；代码题对测试用例写特判；数学题输出多个答案骗宽松的解析器；谄媚用户。对策：KL 正则与早停、严格且确定的 verifier、隐藏测试、长度惩罚或长度无关的归一化、RM 集成、用新策略样本重训 RM、人工抽检 rollout。

## ORM 和 PRM 的区别？为什么 R1 没用 PRM？

ORM 给整条回答一个分，便宜但稀疏；PRM 给每个推理步骤打分，稠密但标注贵（PRM800K 人工标、Math-Shepherd 用 MC rollout 自动估）。PRM 做 best-of-N 重排效果好。DeepSeek-R1 报告 PRM 的问题：步骤难定义、中间步骤对错难判断、引入神经 RM 后会被 hack 且需反复重训，所以用规则奖励。

## 异步 RL 的 staleness 怎么控制？

one-step-off 让生成与训练重叠，数据落后 1 个版本；partial rollout 会让一条轨迹混合多个版本；fully async 落后更多。控制手段：设最大版本差，超出就丢弃；用 IS 权重从行为策略修正到 $\pi_{\text{old}}$（decoupled PPO）；在一个同步周期内固定 $\pi_{\text{old}}$；监控被丢弃样本的比例和平均 staleness。

## 训练中途 reward 崩了，怎么排查？

先看崩之前的指标：grad_norm 尖峰、entropy 异常、response 长度暴涨、`pg_clipfrac` 升高、训推 mismatch KL 变大。依次排查：训推不一致（开 rollout correction、对齐温度与精度）、更新太猛（降 lr、减少 mini-batch 数和 epoch）、奖励被 hack（抽样看 rollout）、数据异常、loss 聚合或数值问题（bf16 溢出）。每次只改一个变量，从最近 checkpoint 复现。

## 训练 rollout 的温度为什么一般取 1.0？

策略梯度要求样本来自被优化的策略。温度不是 1 时采样分布变成 $\pi^{1/T}$ 归一化，若算 logprob 时不用同一温度，就多了一层 off-policy 偏差。取 1.0 同时保证探索。verl 会把 rollout 温度传给计算 logprob 的阶段；验证时再用低温或 greedy。
