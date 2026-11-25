---
title: 一个 PPO step 里发生了什么
summary: 跟着 V1 trainer 的 _step_once 走一遍：生成、打分、算 log-prob、算优势、更新，以及 batch / mini-batch / micro-batch 三层尺寸怎么换算。
tags: verl, PPO, GRPO, 训练循环
order: 20
---

## 入口与 trainer 模式

`python -m verl.trainer.main_ppo` 启动后，`TaskRunnerV1` 作为远程 driver 初始化 TransferQueue、建 WorkerGroup、挑选 trainer。2026-11 起主干**只保留 V1 trainer**（V0 的 `RayPPOTrainer` 已移除，one-step-off / fully-async 两个老的异步方案挪到 `verl-recipe`，固定在 v0.9.1）。

`trainer.v1.trainer_mode` 三选一：

| 模式 | 训练与推理 | partial rollout | 典型用途 |
|---|---|---|---|
| `sync` | 共置，时分复用 | 关 | 标准 on-policy PPO / GRPO |
| `colocate_async` | 共置 | 开 | 想重叠一部分生成和训练，但资源固定 |
| `separate_async` | 分池部署，训练侧空闲时可切去做 rollout | 开 | 长尾严重、追求吞吐的大规模 agent RL |

## `_step_once`：9 个阶段

下面是 `verl/trainer/ppo/v1/trainer_base.py` 里 `_step_once` 的真实顺序（括号里是 `timing_raw` 的 key，日志里看到的耗时就按这些名字统计）：

<div class="flow"><span class="f-infra">1 采样 batch<small>gen</small></span><i>→</i><span>2 奖励<small>reward</small></span><i>→</i><span>3 负载均衡<small>balance</small></span><i>→</i><span>4 old log-prob<small>old_log_prob</small></span><i>→</i><span>5 ref log-prob<small>ref</small></span><i>→</i><span>6 价值<small>values</small></span><i>→</i><span class="f-accent">7 优势<small>adv</small></span><i>→</i><span class="f-rl">8 更新 critic<small>update_critic</small></span><i>→</i><span class="f-rl">9 更新 actor<small>update_actor</small></span></div>

1. **采样（gen）**：从 replay buffer 取一个 batch。sync 模式下就是刚生成完的数据；async 模式下可能混有旧策略生成的轨迹，超过 `sampler.max_off_policy_threshold`（默认 8 个版本）的按 `max_off_policy_strategy: drop` 丢掉。
2. **奖励（reward）**：规则函数（`custom_reward_function.path` + `name`，默认函数名 `compute_score`）或奖励模型打分。奖励可以在 rollout 阶段由 RewardLoop 异步算好，这一步就跳过。
3. **负载均衡**：按 `24576·s + s²` 估计每条序列的计算量，用 `get_seqlen_balanced_partitions` 把 batch 重排成各 DP rank 工作量接近的几份（Karmarkar–Karp 差分法）。
4. **old_log_prob**：用**当前 actor 参数**对 rollout 出来的 response 重算一遍 log-prob，作为 PPO ratio 的分母 $\pi_{\text{old}}$。
5. **ref**（可选）：参考模型 log-prob。只有 `actor.use_kl_loss=True` 或 `algorithm.use_kl_in_reward=True` 时才建 ref。
6. **values**（可选）：只有用 critic（GAE）时才有。
7. **adv**：在 driver 上算优势，按 `algorithm.adv_estimator` 分派（gae / grpo / rloo / reinforce_plus_plus …，见“算法与损失”一篇）。
8. **update_critic**（可选）。
9. **update_actor**：`trainer.critic_warmup <= global_steps` 时才更新 actor，让 critic 先热身几步。

之后 `on_step_end` 里做**权重同步**（`checkpoint_manager.update_weights`），sync 模式下再让 rollout 副本 sleep / wake_up。

> [!Q]
> **为什么要重算 old_log_prob，不直接用 vLLM 生成时返回的 log-prob？**
> 两个原因。(1) 推理引擎和训练引擎的数值不一致：vLLM 常用 bf16 kernel、不同的 attention 实现和采样后处理，算出来的 log-prob 和 FSDP 前向不同，直接用会把 ratio 搞偏；(2) PPO 的 ratio 定义在“训练引擎眼中的旧策略”上，用同一套计算图才能保证第一个 mini-batch 的 ratio 恰好是 1。差异本身可以用 rollout correction（TIS / 拒绝采样）补偿，`rollout_correction.bypass_mode` 则是直接拿 rollout log-prob 当 old 的省算力做法。

## 三层 batch 尺寸

这是 verl 面试里出现频率最高的配置题。

| 配置 | 单位 | 含义 |
|---|---|---|
| `data.train_batch_size` | prompt | 一个 step 采样多少个 prompt |
| `actor_rollout_ref.rollout.n` | — | 每个 prompt 生成几条 response（GRPO 的组大小） |
| `actor.ppo_mini_batch_size` | prompt（全局） | 一次 optimizer step 用多少个 prompt 的数据；内部乘以 `rollout.n` 变成 response 数 |
| `actor.ppo_micro_batch_size_per_gpu` | response（每卡） | 一次前向/反向的样本数，梯度累积用，只影响显存不影响算法 |
| `actor.ppo_epochs` | — | 同一批数据过几遍 |
| `actor.use_dynamic_bsz` + `ppo_max_token_len_per_gpu` | token（每卡） | 按 token 预算动态切 micro-batch，取代固定的 micro batch size |

例子：`train_batch_size=512, rollout.n=8, ppo_mini_batch_size=128, ppo_epochs=1`，8 卡 DP。

- 每个 step 生成 $512\times 8=4096$ 条 response；
- 每个 mini-batch 有 $128\times 8=1024$ 条，每卡 128 条；
- 一个 step 做 $512/128 \times 1 = 4$ 次 optimizer step；
- 若 `ppo_micro_batch_size_per_gpu=16`，每卡每个 mini-batch 累积 $128/16=8$ 次梯度。

> [!KEY]
> 一个 step 内 optimizer 更新次数 = `train_batch_size / ppo_mini_batch_size × ppo_epochs`。等于 1 时完全 on-policy（ratio 恒为 1，clip 不生效）；大于 1 时，从第二个 mini-batch 开始策略已经变了，clip 才开始起作用。

> [!WARN]
> `ppo_micro_batch_size`（不带 `_per_gpu`）已废弃，和 `_per_gpu` 同时设置会直接报错。`separate_async` 模式还要求 `train_batch_size == parameter_sync_step × ppo_mini_batch_size`。

## GRPO 的组是怎么保持在一起的

- 每个 prompt 被采样时分配一个 `uid`；`rollout.n` 条 response 共享这个 uid（V1 里 key 是 `{uid}_{session_id}_{index}`）。
- 计算 GRPO 优势时按 uid 分组求均值和标准差，所以**组必须完整**。async 模式下如果组里有失败的轨迹，`sync_refill_failed_groups` 决定是否同步补齐。
- 负载均衡、dynamic bsz 只是重排样本到不同 rank，不影响优势计算，因为优势在 driver 上先算好了。

## GRPO 的典型配置

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_batch_size=512 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  algorithm.use_kl_in_reward=False \
  trainer.v1.trainer_mode=sync
```

GRPO 不需要 critic，所以第 6、8 步跳过；KL 放在 actor loss 里（`use_kl_loss`）而不是 reward 里。

> [!Q]
> **verl 的一个 step 里，哪一步最慢？怎么优化？**
> 通常是 gen（rollout），长 CoT 下能占一半以上时间，而且被最长那条 response 拖住（长尾）。优化方向：提高推理吞吐（更大 TP、`gpu_memory_utilization`、CUDA Graph、前缀缓存）；限制 `max_response_length`；异步/partial rollout 把长尾挪到下一个 step；训练侧用 remove padding + dynamic bsz 减少 padding 浪费。

> [!Q]
> **`critic_warmup` 有什么用？**
> GAE 的优势依赖 critic，刚初始化的 critic（通常从 RM 或 actor 初始化）估值很不准，直接拿去训 actor 会引入很大偏差。先只更新 critic 若干 step，等价值估计稳定后再开始更新 actor。

> [!SRC]
> 训练循环：`verl/trainer/ppo/v1/trainer_base.py` 的 `_fit` / `step` / `_step_once`；三种模式分别在 `trainer_sync.py`、`trainer_colocate_async.py`、`trainer_separate_async.py`；回放缓冲区 `replay_buffer.py`。
