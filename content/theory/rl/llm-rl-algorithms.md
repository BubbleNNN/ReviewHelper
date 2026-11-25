---
title: LLM-RL 算法族谱
summary: GRPO、Dr.GRPO、DAPO、RLOO、REINFORCE++、ReMax、GSPO、CISPO 各自改了 PPO 的哪一块，以及在 verl 里怎么开。
tags: RL, GRPO, DAPO, verl
order: 40
---

## 一张图看清改动点

LLM 上的 RL 算法几乎都是在 PPO 的四个部件上做替换：

<div class="flow"><span class="f-accent">① 优势/baseline<small>GAE·组均值·留一·greedy</small></span><i>+</i><span class="f-rl">② 比值与 clip<small>token·序列·非对称·截断 IS 权重</small></span><i>+</i><span>③ KL 放哪<small>reward·loss·不要</small></span><i>+</i><span class="f-infra">④ loss 聚合<small>token-mean·seq-mean·常数归一</small></span></div>

去掉 critic 的动机很直接：critic 和 actor 同等规模，显存和算力翻倍；而且在稀疏 outcome reward 下，token 级 value 很难学准。用“同一 prompt 采多条”做 baseline 是最便宜的替代。

## GRPO

DeepSeekMath（Shao et al., 2024）提出。对每个 prompt 采 $G$ 条回答，奖励 $r_1..r_G$，**同一条回答的所有 token 共享一个优势**：

$$\hat A_i = \frac{r_i - \operatorname{mean}(r_{1..G})}{\operatorname{std}(r_{1..G}) + \varepsilon}$$

目标沿用 PPO-clip，但 KL 不放进 reward，而是直接加在 loss 上（用 k3 估计量）：

$$\mathcal J = \frac1G\sum_{i}\frac{1}{|o_i|}\sum_t\Big[\min\big(r_{i,t}\hat A_i,\ \operatorname{clip}(r_{i,t})\hat A_i\big) - \beta\, D_{\text{KL}}^{(k3)}\Big]$$

verl 实现细节（`compute_grpo_outcome_advantage`）：

- 按 `index`（prompt uid）分组；组内只有 1 条时 mean=0、std=1；
- std 用 `torch.std`，**默认无偏（除以 $n-1$）**，与 numpy 默认的有偏 std 不同，手写时要注意；
- $\varepsilon = 10^{-6}$；`norm_adv_by_std_in_grpo=False` 时只减均值；
- 配置：`algorithm.adv_estimator=grpo`、`actor_rollout_ref.rollout.n>1`、`actor.use_kl_loss=True`、`kl_loss_type=low_var_kl`（k3）。

> [!WARN]
> 组内全对或全错时 $r_i - \text{mean} = 0$，整组优势为 0，这些样本**不产生梯度**，但在 token-mean 下仍计入分母，把有效梯度稀释了。这是 DAPO 做 dynamic sampling 的动机。动手：exercises/model/grpo_advantage。

## Dr.GRPO

“Understanding R1-Zero-Like Training”（Liu et al., 2025）指出 GRPO 目标里两个偏置：

1. **长度偏置**：$\frac{1}{|o_i|}$ 让每条回答的 token 平均。对正优势，短回答每个 token 的梯度更大（鼓励短的对的）；对负优势，长回答每个 token 被罚得更轻（纵容长的错的），于是错误回答越写越长。
2. **难度偏置**：除以组内 std 后，std 很小的题（几乎全对或几乎全错）被放大权重。

修正：不除 std，loss 用“token 求和 / 常数”聚合。verl 配置：`algorithm.norm_adv_by_std_in_grpo=False`、`actor.loss_agg_mode=seq-mean-token-sum-norm`（可配 `loss_scale_factor` 为常数，默认取 response 最大长度）、`use_kl_loss=False`。

## DAPO

字节 Seed 与清华 AIR 的工作（arXiv 2503.14476），在 Qwen2.5-32B base 上报告 AIME 2024 50 分。四个改动，外加去掉 KL：

| 改动 | 做法 | verl 配置 |
|---|---|---|
| Clip-Higher | $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$，让低概率 token 能涨上来，缓解熵塌缩 | `actor.clip_ratio_low/high` |
| Dynamic Sampling | 丢弃 acc 全 0 或全 1 的组，不够就继续采样补满 batch | `algorithm.filter_groups.enable/metric/max_num_gen_batches`，`data.gen_batch_size` |
| Token-level loss | 整个 batch 所有 token 平均（token-mean），长回答按 token 数获得相应权重 | `actor.loss_agg_mode=token-mean` |
| Overlong reward shaping | 长度超过 $L_{\max}-L_{\text{cache}}$ 后线性扣分，超过 $L_{\max}$ 扣满 | `reward_model.overlong_buffer.enable/len/penalty_factor` |

overlong 惩罚（verl 实现）：

$$r_{\text{overlong}} = \min\Big(-\frac{|y| - (L_{\max} - L_{\text{buf}})}{L_{\text{buf}}}\cdot \text{factor},\ 0\Big)$$

> [!NOTE]
> verl 文档里 DAPO 复现：完整 DAPO 52%，去掉 dynamic sampling 50%，再去掉 token-level loss 44%（AIME 2024，16×8 卡）。论文里还提到 overlong filtering（直接 mask 掉被截断样本），但最佳结果没开它。

## RLOO

“Back to Basics”（Ahmadian et al., 2024）。每个 prompt 采 $k$ 条，用**其他** $k-1$ 条的平均做 baseline：

$$\hat A_i = r_i - \frac{1}{k-1}\sum_{j\ne i} r_j = \frac{k}{k-1}\big(r_i - \bar r\big)$$

baseline 与当前样本独立，所以无偏；与“减组均值”只差一个常数因子 $\frac{k}{k-1}$。原文主张在 RLHF 里 PPO 的 clip 和 critic 都可以省掉，直接 REINFORCE。verl：`adv_estimator=rloo`（另有 `rloo_vectorized`）。

## REINFORCE++ 与 baseline 版

REINFORCE++（Hu, 2025, arXiv 2501.03262）的要点：不要 critic；KL 作为**token 级惩罚**放进 reward；优势用**全局 batch** 标准化而不是组内标准化，作者认为组内标准化容易过拟合简单题。

- `reinforce_plus_plus`：对 token 级 reward（含 KL 惩罚）做折扣累加得到 return-to-go，再整 batch `masked_whiten`。
- `reinforce_plus_plus_baseline`：先减组均值，再整 batch `masked_whiten`，适合 outcome reward + 每题多采样。

## ReMax

ReMax（Li et al., 2023）：baseline 是同一 prompt **greedy 解码**回答的奖励，$\hat A = r(y) - r(y_{\text{greedy}})$。只需要一条额外的 greedy 生成，省掉 critic。verl：`adv_estimator=remax`，trainer 会额外生成 greedy 样本并把奖励作为 `reward_baselines` 传入。

## GSPO

Qwen 团队（Zheng et al., 2025, arXiv 2507.18071）。核心论点：重要性比值应该和奖励同一粒度。奖励是序列级的，token 级比值每个 token 只有一个样本，噪声大，长序列上会累积，MoE 模型上尤其不稳（路由变化导致 token 比值剧烈波动）。GSPO 用**长度归一化的序列比值**：

$$s_i(\theta) = \Big(\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\Big)^{1/|y_i|} = \exp\Big(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta(y_{i,t}\mid\cdot)}{\pi_{\text{old}}(y_{i,t}\mid\cdot)}\Big)$$

clip 也在序列级做，整条回答要么都有梯度、要么都被截断。因为是几何平均，clip 范围要小得多（论文取 $3\times10^{-4}$ / $4\times10^{-4}$ 量级，待核实）。verl：`actor.policy_loss.loss_mode=gspo`。

## CISPO

MiniMax-M1 技术报告（2025）提出。PPO 的 clip 会让越界 token 的梯度直接变成 0，而一些低概率、承载“反思/转折”的关键 token 恰恰最容易越界。CISPO 改成**截断重要性权重本身**并 stop-gradient，所有 token 都保留 $\log\pi$ 的梯度：

$$\mathcal J = \mathbb E\Big[\operatorname{sg}\big(\operatorname{clip}(r_t, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\big)\,\hat A_t\,\log\pi_\theta(a_t\mid s_t)\Big]$$

verl：`loss_mode=cispo`。具体 $\epsilon$ 取值：待核实。

## 对比表

| 算法 | critic | 优势 / baseline | ratio 与 clip | KL | loss 聚合 | verl 开关 |
|---|---|---|---|---|---|---|
| PPO | 要 | GAE | token，对称 clip | reward 中（经典 RLHF） | token-mean（verl 默认） | `adv_estimator=gae` |
| GRPO | 不要 | 组均值 / 组 std | token，对称 clip | loss 中，k3 | 原文 seq-mean-token-mean | `grpo` |
| Dr.GRPO | 不要 | 组均值，不除 std | token | 去掉 | token 求和 / 常数 | `grpo` + `norm_adv_by_std_in_grpo=False` |
| DAPO | 不要 | 组标准化 + 动态采样 | token，非对称 clip | 去掉 | token-mean | `grpo` + `clip_ratio_high` + `filter_groups` |
| RLOO | 不要 | 留一均值 | 原文无 clip | 视实现 | 视实现 | `rloo` |
| REINFORCE++ | 不要 | return 全局标准化 | token，clip | reward 中，token 级 | token-mean | `reinforce_plus_plus` |
| ReMax | 不要 | greedy 回答奖励 | token | 视实现 | 视实现 | `remax` |
| GSPO | 不要 | 组标准化 | **序列级**几何平均 | 视实现 | 序列级 | `loss_mode=gspo` |
| CISPO | 不要 | 组标准化 | 截断 IS 权重，不截梯度 | 视实现 | token | `loss_mode=cispo` |

> [!Q]
> **GRPO 为什么不需要 critic？代价是什么？**
>
> 用同 prompt 的 $n$ 条回答奖励均值近似 $V(x)$，省掉一个与 actor 同规模的模型。代价：每个 prompt 要采 $n$ 条（rollout 成本 ×n）；所有 token 共享序列级优势，没有 token 级信用分配；组内全对 / 全错时没有信号；std 归一化引入难度偏置。

> [!Q]
> **GRPO 的 std 归一化有什么问题？**
>
> 一是难度偏置：几乎全对的组 std 很小，少数错误样本的负优势被放大成很大的值，梯度被简单题主导。二是 baseline 依赖了当前样本本身，估计不再严格无偏。Dr.GRPO 直接去掉 std；REINFORCE++ 改成全局 batch 标准化，让尺度在 batch 内统一。

> [!Q]
> **token-mean 和 seq-mean-token-mean 哪个好？**
>
> seq-mean-token-mean 给每条回答相同总权重，长回答里每个 token 的权重被稀释，于是长错误回答受罚更轻（长度偏置）；token-mean 让每个 token 权重相同，长回答总权重更大。DAPO 实验里 token-mean 更好，verl 的 GRPO 示例也默认 token-mean；想彻底去掉长度相关的归一化就用 seq-mean-token-sum-norm。

> [!SRC]
> 优势估计器注册在 `verl/trainer/ppo/core_algos.py` 的 `AdvantageEstimator` 枚举里（gae、grpo、reinforce_plus_plus、reinforce_plus_plus_baseline、remax、rloo、opo、grpo_passk、gpg、rloo_vectorized、grpo_vectorized、optimal_token_baseline、tir_optimal_token_baseline、gdpo），用 `@register_adv_est` 扩展；policy loss 用 `@register_policy_loss` 注册（vanilla、gspo、sapo、gpg、clip_cov、kl_cov、geo_mean、dro、cispo、dppo_tv、dppo_kl 等）。

## 速记

- 去 critic 的核心是换 baseline：组均值（GRPO）、留一（RLOO）、greedy（ReMax）、全局标准化（REINFORCE++）。
- Dr.GRPO = 不除 std + 常数归一化；DAPO = clip-higher + 动态采样 + token-mean + overlong 惩罚 − KL。
- GSPO 把 ratio 和 clip 提到序列级；CISPO 截断权重但保留所有 token 的梯度。
- 动手：exercises/model/grpo_advantage、exercises/model/ppo_loss。
