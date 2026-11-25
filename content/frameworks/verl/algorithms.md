---
title: 算法与损失：优势估计、策略损失、KL、聚合方式
summary: 对照 core_algos.py 把 verl 里的 adv_estimator、policy loss、KL 两种放法和 loss_agg_mode 一次讲清，附默认值。
tags: verl, PPO, GRPO, KL, 源码
order: 30
---

verl 的算法都集中在 `verl/trainer/ppo/core_algos.py`，用两个注册表组织：`register_adv_est`（优势估计）和 `register_policy_loss`（策略损失）。下面的默认值都来自 `_generated_ppo_trainer.yaml`（2026-11 主干）。

## 优势估计：`algorithm.adv_estimator`

| key | 需要 critic | 怎么算 | 一句话 |
|---|---|---|---|
| `gae`（默认） | 是 | 反向递推 $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$，再 `masked_whiten` | 经典 PPO；`gamma=lam=1.0` |
| `grpo` | 否 | 组内 $(r_i-\bar r)/(\sigma+\epsilon)$，$\sigma$ 用 `torch.std`（无偏，除以 $n-1$） | `norm_adv_by_std_in_grpo=False` 即 Dr.GRPO |
| `rloo` | 否 | $r_i-\frac{1}{n-1}\sum_{j\ne i}r_j$ | leave-one-out 基线，无偏 |
| `reinforce_plus_plus` | 否 | 折扣回报后在**整个 batch** 上 whiten | 不分组，全局归一化 |
| `reinforce_plus_plus_baseline` | 否 | 先减组均值，再在整个 batch 上 whiten | 组内去偏 + 全局归一化 |
| `remax` | 否 | 回报减去贪心解码 response 的奖励 | 需要额外一次 greedy 生成 |
| `opo`、`gpg`、`grpo_passk`、`gdpo`、`optimal_token_baseline` … | 否 | 各论文的变体 | 面试知道有就行 |

几个细节：

- 只有一条 response 的组：GRPO 把均值设 0、标准差设 1，即优势 = 原始奖励；RLOO 不做修改。
- GRPO 的标量优势会广播到 response 的每个 token：`scores.unsqueeze(-1) * response_mask`。
- 动手：`exercises/model/gae`、`exercises/model/grpo_advantage`。

> [!Q]
> **GRPO 的 std 归一化有什么副作用？**
> (1) 难度偏置：全对或全错附近的组 std 很小，优势被放大，模型过度关注“几乎都会/几乎都不会”的题；(2) 全对或全错的组 std=0，优势恒为 0，不贡献梯度，浪费样本。Dr.GRPO 去掉 std 归一化；DAPO 用动态采样把这些组直接过滤掉再补采。

## 策略损失：`actor.policy_loss.loss_mode`

默认 `vanilla`，就是带 clip-higher 和 dual-clip 的 PPO：

$$
L^{\text{clip}}_t=\max\big(-A_t\,\rho_t,\;-A_t\,\mathrm{clip}(\rho_t,1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\big),\qquad \rho_t=e^{\log\pi_\theta-\log\pi_{\text{old}}}
$$

$$
L_t=\begin{cases}\min\big(L^{\text{clip}}_t,\;-A_t\,c\big) & A_t<0\\ L^{\text{clip}}_t & A_t\ge 0\end{cases}
$$

| 配置 | 默认 | 作用 |
|---|---|---|
| `clip_ratio` | 0.2 | 对称 clip 的 $\epsilon$ |
| `clip_ratio_low` / `clip_ratio_high` | 0.2 / 0.2 | 非对称 clip；DAPO 的 clip-higher 把 high 调到 0.28 一类的值，给低概率 token 更多上涨空间 |
| `clip_ratio_c` | 3.0 | dual-clip 下界，必须 > 1；只作用于 $A<0$，防止 ratio 很大时负优势带来无界的损失 |

实现上还有两个保护：$\log\rho$ 先 clamp 到 $[-20,20]$ 再取 exp；同时输出三个监控指标 `pg_clipfrac`（被 clip 的 token 比例）、`ppo_kl`（$\mathbb E[\log\pi_{\text{old}}-\log\pi_\theta]$）、`pg_clipfrac_lower`（被 dual-clip 截住的比例）。

其他 loss_mode：

| loss_mode | 要点 |
|---|---|
| `gspo` | 序列级 ratio：$\exp\big(\frac1{|y|}\sum_t(\log\pi_\theta-\log\pi_{\text{old}})\big)$，用 stop-gradient 技巧让每个 token 的梯度仍然是自己的；默认聚合 `seq-mean-token-mean` |
| `cispo` | 把 clip 后的 ratio 当常数权重：$-\text{sg}(\mathrm{clip}(\rho))\,A\,\log\pi_\theta$，被 clip 的 token 仍有梯度 |
| `clip_cov` / `kl_cov` | 针对熵坍缩：只对 log-prob 与优势协方差最大的少数 token 做 clip / 加 KL |
| `gpg`、`sapo`、`dppo_tv`、`dppo_kl`、`geo_mean`、`dro` | 其他论文的变体 |
| `bypass_mode` | 直接用 rollout 的 log-prob 当 $\pi_{\text{old}}$，配合 rollout correction 使用 |

动手：`exercises/model/ppo_loss`（和 `compute_policy_loss_vanilla` + `agg_loss` 对齐）。

> [!Q]
> **为什么 dual-clip 只管 $A<0$？**
> $A>0$ 时 PPO 的 max 已经在 $\rho>1+\epsilon$ 处把损失截成常数。$A<0$ 时，如果 $\rho$ 很大（新策略大幅提高了一个坏动作的概率），$-A\rho$ 会无限增大，单个 token 就可能主导梯度。dual-clip 把它截在 $-A\cdot c$。

## KL：放 reward 里，还是放 loss 里

| | KL in reward | KL in loss |
|---|---|---|
| 开关 | `algorithm.use_kl_in_reward=True` | `actor.use_kl_loss=True` |
| 公式 | $r_t \leftarrow r_t-\beta\,(\log\pi_{\text{old}}-\log\pi_{\text{ref}})$ | $L \mathrel{+}= \beta\cdot \mathrm{KL}(\pi_\theta,\pi_{\text{ref}})$ |
| 估计量 | `algorithm.kl_penalty`，默认 `kl`（k1） | `actor.kl_loss_type`，默认 `low_var_kl`（k3） |
| 系数 | `algorithm.kl_ctrl.kl_coef=0.001`，`type: fixed`（或 `adaptive`） | `actor.kl_loss_coef=0.001` |
| 典型用法 | 经典 RLHF-PPO（InstructGPT） | GRPO / DAPO 系 |

两者的本质区别：放 reward 里，KL 先进入回报和优势，经过 GAE 再影响梯度，**是被当成环境奖励的一部分来优化**；放 loss 里，KL 直接对 $\pi_\theta$ 求梯度，是一个正则项，不会被优势归一化稀释。

估计量（`kl_penalty_forward`），记 $x=\log\pi-\log\pi_{\text{ref}}$：

| 名字 | 公式 | 性质 |
|---|---|---|
| `kl` / `k1` | $x$ | 无偏，方差大，单个样本可为负 |
| `mse` / `k2` | $\tfrac12 x^2$ | 有偏，但它的**梯度**期望等于真实 KL 的梯度 |
| `low_var_kl` / `k3` | $e^{-x}+x-1$，clamp 到 $[-10,10]$ | 无偏、恒非负、方差小 |
| `abs` | $\lvert x\rvert$ | 启发式 |

加 `+` 后缀（如 `k3+`）启用 straight-through：前向数值用 k3，反向梯度用 k2：`k2 - k2.detach() + k3.detach()`。理由是 k1、k3 的期望等于 KL，但它们梯度的期望不等于 KL 的梯度。

AdaptiveKLController（Ziegler et al. 2019）的更新：

$$
e=\mathrm{clip}\Big(\frac{\mathrm{KL}_{\text{cur}}}{\mathrm{KL}_{\text{target}}}-1,\,-0.2,\,0.2\Big),\qquad \beta\leftarrow\beta\,\Big(1+e\cdot\frac{n_{\text{steps}}}{\text{horizon}}\Big)
$$

默认 `target_kl=0.1, horizon=10000`。

动手：`exercises/model/kl_estimators`。

## Loss 聚合：`actor.loss_agg_mode`

逐 token 损失矩阵 `loss_mat (B, T)` 怎么变成一个标量，决定了长短 response 的相对权重。verl 的实现保证**在不同 DP 切法下结果一致**：每个 rank 先求本地和，除以**全局**的 token 数或序列数，再乘 `dp_size`（因为 DDP/FSDP 的梯度同步是取平均）。

| 模式 | 公式（单卡视角） | 长度偏置 |
|---|---|---|
| `token-mean`（默认） | $\frac{\sum_{i,t} m_{it}\ell_{it}}{\sum_{i,t} m_{it}}$ | 每个 token 权重相同，长 response 总贡献更大 |
| `token-sum` | $\sum_{i,t} m_{it}\ell_{it}$ | 尺度随 batch token 数变化 |
| `seq-mean-token-mean` | $\frac1B\sum_i \frac{\sum_t m_{it}\ell_{it}}{\sum_t m_{it}}$ | 原版 GRPO：每条序列权重相同，长序列里单个 token 被稀释；长 CoT 下不稳定 |
| `seq-mean-token-sum` | $\frac1B\sum_i \sum_t m_{it}\ell_{it}$ | 长序列权重大 |
| `seq-mean-token-sum-norm` | 上式再除以常数 `loss_scale_factor`（默认取 $T_{\max}$） | Dr.GRPO：用常数归一化，消掉“按长度除”带来的偏置 |

> [!WARN]
> `token-mean` 在多卡下不能各算各的平均再求均值：每卡 token 数不同，那样等于给每张卡等权。verl 要求传入全局 `batch_num_tokens`，否则 `dp_size>1` 时直接报错。动手题 `exercises/model/ppo_loss` 只考虑 `dp_size=1`。

## 其他常用开关

- `actor.entropy_coeff`：默认 0。熵奖励在 LLM 上容易让输出变乱，现在更多用 clip-higher、Clip-Cov 来对抗熵坍缩。
- `critic.cliprange_value`：value clip，$\max\big((V-R)^2,(\mathrm{clip}(V, V_{\text{old}}\pm\epsilon)-R)^2\big)$，前面乘 0.5。
- DAPO 的动态采样：`algorithm.filter_groups.enable=True`，`metric: reward`，丢掉组内奖励全相同的组并补采，最多补 `max_num_gen_batches` 次（0 表示不限）。
- Rollout correction（训推不一致修正）：`algorithm.rollout_correction.rollout_is` 取 `token`（有偏、低方差）或 `sequence`（无偏、高方差），默认阈值 `rollout_is_threshold=2.0`（截断重要性采样 TIS）；`rollout_rs` 做拒绝采样（`token_k1`/`token_k2`/`token_k3`、`seq_sum_k*`、`seq_mean_k*` 等），把偏离太大的样本直接 mask 掉；要求 rollout 时 `calculate_log_probs=True`。配置类里还有 IcePop、几何平均 RS 等预设。

> [!Q]
> **想实现一个新的优势估计或损失，在 verl 里怎么接？**
> 写一个函数，用 `@register_adv_est("my_adv")` 或 `@register_policy_loss("my_loss")` 注册，签名对齐已有函数（拿到 `token_level_rewards`、`response_mask`、`index`、`config` 等），然后在配置里设 `algorithm.adv_estimator=my_adv` 或 `actor.policy_loss.loss_mode=my_loss`。不用改 trainer。

> [!SRC]
> `compute_policy_loss_vanilla`、`agg_loss`、`kl_penalty`、`compute_value_loss`、`AdaptiveKLController` 都在 `verl/trainer/ppo/core_algos.py`；rollout correction 在 `verl/trainer/ppo/rollout_corr_helper.py`。
