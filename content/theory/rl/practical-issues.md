---
title: LLM-RL 实践中的坑
summary: 训推不一致与 rollout correction、异步带来的 staleness、loss_agg_mode 与长度偏置、零优势组、过长截断、稀疏奖励，以及调参和监控清单。
tags: RL, 训练稳定性, verl
order: 70
---

## 训练-推理不一致

LLM-RL 里同一组权重有两套实现：rollout 用 vLLM / SGLang（追求吞吐），训练用 FSDP / Megatron（追求可微）。**同一个 token 两边算出的 logprob 并不相同**：

- kernel 不同（attention、fused op 的归约顺序），TP 切分方式不同；
- 精度不同（bf16 累加、rollout 用 FP8 量化）；
- batch 组成不同导致结果不同（kernel 非 batch-invariant）；
- 采样后处理（温度、top-p 截断）是否计入 logprob；
- MoE 中微小数值差异会让路由选到不同专家，差异被放大。

于是存在三个策略：

| 记号 | 来源 | 用途 |
|---|---|---|
| $\pi_{\text{rollout}}$ | 推理引擎，真正产生样本的行为策略 | 需要 `rollout.calculate_log_probs=True` 才拿得到它的 logprob |
| $\pi_{\text{old}}$ | 训练引擎用同一份权重重算（`old_log_prob` 阶段） | PPO ratio 的锚点 |
| $\pi_\theta$ | 正在更新的参数 | 求梯度 |

很多实现默认 $\pi_{\text{old}} = \pi_{\text{rollout}}$，但数值上并不成立：表面上 on-policy，实际是 off-policy。不一致随训练积累，长序列、MoE、低精度 rollout 下尤其严重，表现为某一步之后 reward 断崖、梯度范数爆炸。

### rollout correction

verl 的 `algorithm.rollout_correction` 把修正拆成两种正交机制：

<div class="cols">
<div><h4>IS 权重（重新加权）</h4><p>decoupled 模式下权重是 \(\pi_{\text{old}}/\pi_{\text{rollout}}\)，乘在 policy loss 上。</p><p><code>rollout_is: token | sequence</code>；<code>rollout_is_threshold: 2.0</code> 是 TIS（截断重要性采样，只截上界）。</p><p>阈值写成 <code>"0.5_5.0"</code> 就是 IcePop：区间外的 token 权重置 0，不改 response_mask。</p></div>
<div><h4>拒绝采样（过滤）</h4><p><code>rollout_rs</code> 按比值把 token 或整条序列从 response_mask 里剔除，例如 <code>token_k1</code>、<code>seq_max_k2</code>、<code>seq_sum_k1</code>。</p><p><code>rollout_rs_threshold</code> 给上下界，如 <code>"0.5_2.0"</code>。</p></div>
</div>

另有 `bypass_mode=True`：直接令 $\pi_{\text{old}} = \pi_{\text{rollout}}$，省掉一次重算 logprob 的前向，PPO 的 ratio 变成 $\pi_\theta/\pi_{\text{rollout}}$。

截断 IS（TIS）的写法：

$$\mathcal L = -\,\mathbb E_t\Big[\min\Big(\frac{\pi_{\text{old}}}{\pi_{\text{rollout}}},\ C\Big)\cdot \text{PPOClip}\big(r_t(\theta), \hat A_t\big)\Big]$$

> [!TIP]
> 工程侧也要缩小差距：LM head 用 fp32、训练和推理用同一采样温度并且算 logprob 时 logits 除以同一温度、对齐 attention kernel、使用 batch-invariant / 确定性 kernel；MoE 可以开 rollout 路由回放（verl `enable_rollout_routing_replay`，让训练前向复用 rollout 时选中的专家）。

> [!Q]
> **既然 verl 会用训练引擎重算 old_log_prob，为什么还需要 rollout correction？**
>
> 重算只解决“PPO ratio 的锚点要和训练引擎一致”，样本仍然是按 $\pi_{\text{rollout}}$ 采的。从 $\pi_{\text{rollout}}$ 到 $\pi_{\text{old}}$ 的分布差没有被修正，梯度是有偏的。IS 权重 $\pi_{\text{old}}/\pi_{\text{rollout}}$ 正是补这一段；截断（TIS）或置零（IcePop）是为了控制方差。

## off-policy 与 staleness

同步 RL 的 GPU 利用率被最长的 rollout 拖住。异步方案用 staleness 换吞吐：

| 方案 | 做法 | 数据有多旧 |
|---|---|---|
| one-step-off | 第 k 步训练与第 k+1 步的生成重叠 | 恰好落后 1 个版本 |
| partial rollout | 权重更新时中断生成，用新权重续写 | 一条轨迹混合多个版本 |
| fully async / separate | 生成与训练在不同 GPU 池上持续运行，经 replay buffer 交换 | 落后若干版本，需设上限 |

对策：限制最大 staleness，超过就丢弃（verl V1 的 ReplayBuffer 会统计被驱逐样本的 staleness）；decoupled PPO，用一个稳定的“近端策略”做 clip 锚点，再用 IS 权重从行为策略修正过来（verl separate_async 会把 $\pi_{\text{old}}$ 权重存到 CPU，保证一个同步周期内各 mini-batch 用同一个 $\pi_{\text{old}}$）。

## loss_agg_mode 与长度偏置

`loss_mat` 是 `(bs, T)` 的逐 token 损失，`agg_loss` 决定怎么变成标量。设一个 mini-batch 有 $B$ 条序列、共 $N$ 个有效 token，第 $i$ 条长 $L_i$：

| 模式 | 计算 | 单个 token 的权重 |
|---|---|---|
| `token-mean`（默认） | 所有有效 token 求平均 | $1/N$ |
| `seq-mean-token-mean` | 每条序列内平均，再对序列平均 | $1/(B L_i)$ |
| `seq-mean-token-sum` | 每条序列内求和，再对序列平均 | $1/B$ |
| `seq-mean-token-sum-norm` | 上一种再除以常数 `loss_scale_factor`（默认是 response 最大长度） | $1/(B H)$ |
| `token-sum` | 全部求和 | $1$ |

- `seq-mean-token-mean`：长序列中每个 token 权重小，长的错误回答受罚轻，正是 Dr.GRPO 说的长度偏置。
- `seq-mean-token-sum`：长序列总权重随长度线性增长，梯度尺度随长度波动。
- `token-mean`：每个 token 平等，DAPO 采用；长回答整体权重更大。
- `seq-mean-token-sum-norm`：每个 token 权重恒定，与长度无关，Dr.GRPO 推荐。

### 为什么要乘 dp_size、除以全局 token 数

```python
if loss_agg_mode == "token-mean":
    loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
```

`batch_num_tokens` 是**整个 mini-batch 在所有 DP rank 上**的有效 token 数。每个 micro-batch 只算自己那部分 token 的和，除以全局 token 数；梯度累积把各 micro-batch 相加；DDP/FSDP 的 all-reduce 会对 rank 求平均（相当于除以 dp_size），所以先乘回 dp_size。最终梯度恰好是“全局所有 token 的平均”，**与 micro-batch 怎么切、每个 rank 分到多少 token 都无关**。

> [!WARN]
> 常见错误：每个 micro-batch 先做 `masked_mean`，再对 micro-batch 取平均。当各 micro-batch 的 token 数不同（动态 batch、长度差异大）时，短 micro-batch 里的 token 会被放大权重，结果随切分方式变化。`seq-mean-*` 模式同理要用全局序列数 `global_batch_size` 做分母；完全被 mask 的序列不计入分母。

> [!SRC]
> `verl/trainer/ppo/core_algos.py::agg_loss`：参数 `dp_size`、`batch_num_tokens`、`global_batch_size`、`loss_scale_factor`；`dp_size>1` 却没传全局 token 数时直接报错。value loss 和 entropy loss 也走同一个函数。

## 零优势组

GRPO / RLOO 类方法中，一个 prompt 的 n 条回答全对或全错，组内优势全为 0：

- 这些样本不贡献梯度，却占用 rollout 算力，在 token-mean 下还稀释有效梯度；
- 随着训练推进，简单题越来越多“全对”，有效 batch 持续缩小。

对策：dynamic sampling（`algorithm.filter_groups`，过滤后不足就继续生成，`max_num_gen_batches` 设上限）；离线按 pass rate 过滤数据，只保留 0 和 1 之间的题；增大 `rollout.n`；课程学习。监控“优势为 0 的组占比”。

## 过长截断

到达 `max_response_length` 的回答没有结束，通常拿不到正确奖励：

- 一个正确但冗长的推理被判 0 分，这是**奖励噪声**，会把模型推向“短”或者推向混乱；
- 对 critic 来说，截断不是真正的终止，严格应 bootstrap $V(s_T)$，多数实现直接当终止。

选择：overlong filtering（把被截断样本从 loss 中 mask 掉）；DAPO 的软惩罚（`reward_model.overlong_buffer`）；分阶段加长上下文；监控 `response_length/clip_ratio`。

## 稀疏奖励与信用分配

outcome reward 下，序列级优势平均分给所有 token，“关键一步”与“废话”同等对待。改进方向：

- 有 critic 的 PPO 提供 token 级优势，但 value 在长 CoT 上难学（需要好的初始化与 warmup）；
- 过程奖励 / PRM（代价与风险见“偏好学习与奖励设计”）；
- 多轮 agent 中给 turn 级奖励；
- 更多样本（增大 n）降低方差。

## 调参与监控清单

常见起点（经验值，不是定论，需按模型和任务调整）：

| 项 | 常见取值 | verl 键 |
|---|---|---|
| actor 学习率 | 1e-6 量级，常数 + 少量 warmup | `actor.optim.lr`（默认 1e-6） |
| critic 学习率 | 1e-5 量级 | `critic.optim.lr`（默认 1e-5） |
| 梯度裁剪 | 1.0 | `actor.grad_clip`（默认 1.0） |
| 每题采样数 | 4~16 | `rollout.n` |
| 训练 rollout 温度 | 1.0，top_p 1.0 | `rollout.temperature` |
| clip | 0.2 / 0.28 | `clip_ratio_low/high` |
| KL | 0.001 或关闭 | `kl_loss_coef` / `use_kl_loss` |
| 熵系数 | 0 | `entropy_coeff` |

监控指标：

| 指标 | 看什么 |
|---|---|
| `critic/score/mean` | 训练奖励，应平稳上升 |
| `response_length/mean`、`response_length/clip_ratio` | 长度暴涨或截断比例升高 |
| `actor/entropy` | 断崖下降 = 熵塌缩，持续上升 = 熵爆炸 |
| `actor/pg_clipfrac`、`actor/ppo_kl` | 更新步长是否过大 |
| `actor/grad_norm` | 尖峰往往先于崩溃出现 |
| `rollout_corr/*` | 训推不一致程度（mismatch KL 等） |
| `critic/vf_explained_var` | critic 是否学到东西（PPO） |
| `perf/throughput`、`timing_s/*` | 哪个阶段是瓶颈 |

> [!Q]
> **训练到一半 reward 突然崩掉，你怎么排查？**
>
> 先看崩之前的信号：grad_norm 尖峰、entropy 异常、response 长度暴涨、clip 比例、训推 mismatch KL。按可能性排查：① 训推不一致（开 rollout correction、对齐精度与温度）；② 学习率或更新步数太大（降 lr、减 epoch / mini-batch 数）；③ 奖励被 hack（抽样看 rollout）；④ 数据问题（某类题全对全错、格式异常）；⑤ 数值问题（bf16 溢出、loss 聚合错误）。每次只改一个变量，从最近的 checkpoint 复现。

> [!Q]
> **为什么训练 rollout 的温度一般取 1.0？**
>
> 策略梯度要求样本来自被优化的策略本身。温度不是 1 时，采样分布是 $\pi^{1/T}$ 的归一化，若算 logprob 时没有用同一温度，就引入了额外的 off-policy 偏差；用同一温度则优化的是“加温后的策略”。1.0 同时保留了足够的探索。验证时再用低温或 greedy。

## 速记

- 三个策略：$\pi_{\text{rollout}}\ne\pi_{\text{old}}$ 是常态；TIS 截上界、IcePop 区间外置零、RS 直接剔除。
- 异步换吞吐，代价是 staleness；限制版本差并做 IS 修正。
- loss 要按全局 token / 序列数归一化，乘 dp_size 抵消 all-reduce 的平均。
- 零优势组 → dynamic sampling；截断 → overlong filtering / 软惩罚。
