---
title: 从 logits 取 token logprob 与熵（含 response 对齐）
difficulty: 易
tags: RL, verl, logprob
order: 24
source: verl/utils/torch_functional.py · logprobs_from_logits / entropy_from_logits
---

RL 训练里 actor 要对 `prompt + response` 整条序列做一次前向，然后取出 **response 部分每个 token 的 log 概率**（算 ratio、KL 都要用），以及每个位置的熵（监控 entropy collapse）。

输入输出都是 `torch.Tensor`（测试用 float64），输出 dtype 与 `logits` 相同：

```python
logprobs_from_logits(logits, labels)     # logits (B, T, V) float, labels (B, T) long -> (B, T)
entropy_from_logits(logits)              # (B, T, V) -> (B, T)，H = -Σ p log p
response_logprobs(logits, input_ids, response_length)   # input_ids (B, T) long -> (B, R)
masked_mean(x, mask)                     # mask: bool，True 的位置参与平均；返回 0-d tensor
```

**对齐规则**：自回归模型第 `t` 个位置的 logits 预测的是第 `t+1` 个 token。`input_ids` 的最后 `R = response_length` 个 token 是 response，所以要用 `logits[:, -R-1:-1]` 去取 `input_ids[:, -R:]` 的概率。

所有计算都要数值稳定（logits 可能到 `1e3` 量级）。可以用 `torch.logsumexp`、`torch.gather` 这类基础算子，但**不要直接调用** `F.cross_entropy`、`F.nll_loss`、`torch.distributions.Categorical`（它们本身就是答案，测试拿它们当对照）。`masked_mean` 要返回 0-d tensor 而不是 python float，因为它之后要当 loss 反传。

<!-- solution -->

### 讲解

**取 logprob**：`log_softmax` 后在 vocab 维 gather：

$$\log\pi(y_t)=z_{t,y_t}-\log\sum_v e^{z_{t,v}}$$

只需要被选中 token 的 logit 和整行的 logsumexp，不必把整个 `log_softmax` 结果存下来。verl 在 GPU 上会尽量用 flash-attn 的 `cross_entropy_loss` 等融合 kernel 来算，避免物化 `(B, T, V)` 的 log 概率张量（$V$ 约 15 万时这个张量非常大）。

**熵**：$H=\text{logsumexp}(z)-\sum_v p_v z_v$。这是 verl `entropy_from_logits` 的写法，比 $-\sum p\log p$ 少一次 log。

**对齐**：`[:, -R-1:-1]` 这个切片是 RL 代码里最常见的 off-by-one 来源。可以这样记：最后一个位置的 logits 预测的是“序列之后的下一个 token”，在训练里用不到，所以切掉。

**易错点**

- 先把 logits 转成 fp32 再算；bf16 的 logsumexp 误差会直接体现在 ratio 上，导致 $\pi_\theta/\pi_{old}$ 明明是同一个策略却不等于 1。
- 带 padding 的 batch 要配 `response_mask`，均值用 `masked_mean`，不能直接 `.mean()`。bool mask 做 `1 - mask` 这类减法会报错，统一先 `mask.to(x.dtype)` 再算。
- `gather` 的 index 必须是 long，而且维数要和 logits 一样：`labels.unsqueeze(-1)` 变成 `(B, T, 1)`，取完再 `squeeze(-1)`。
- temperature：如果 rollout 用了温度 $T$，训练时算 logprob 也要先把 logits 除以 $T$，否则策略分布对不上（verl 里会把 rollout 的 temperature 传给 actor 计算 logprob）。

> [!Q]
> 为什么 rollout 引擎（vLLM）返回的 logprob 和训练框架重算的不一样？
> kernel 实现不同（attention、matmul 的累加顺序）、精度不同（bf16/fp8）、batch 组合不同导致的非确定性，都会让两边数值有差异。这就是“训练-推理不一致”，verl 用 `rollout_correction`（TIS 等）做重要性修正。

```python
# torch
logp = torch.log_softmax(logits.float(), -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
ent = torch.logsumexp(logits, -1) - (logits.softmax(-1) * logits).sum(-1)
```
