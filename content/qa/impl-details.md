---
title: 实现与数值细节
summary: 手撕代码时面试官最爱追问的数值稳定性、形状和对齐问题。
tags: 代码
order: 60
---

## 手写 softmax 为什么要先减去最大值？会改变结果吗？

不会改变结果：分子分母同乘 $e^{-c}$ 抵消了，$\text{softmax}(x)=\text{softmax}(x-c)$。

减去 $c=\max x$ 之后每一项 $e^{x_i-c}\le 1$，不会上溢；最大那一项是 $e^0=1$，分母至少为 1，也不会下溢到 0。不减的话 fp32 下 $x>88$ 就会得到 `inf`，结果变成 `nan`。

动手：`exercises/model/softmax_logsumexp`

## log_softmax 为什么不能写成 log(softmax(x))？

softmax 把很小的概率下溢成 0 之后，再取 log 就是 `-inf`，梯度变成 `nan`。

正确写法是 $x_i - \text{logsumexp}(x)$，全程在 log 空间里算，概率是 $e^{-5000}$ 的 token 也能得到准确的有限值 $-5000$。PPO/GRPO 里的 token logprob 都应该这样算。

## bf16 和 fp16 有什么区别？softmax / logsumexp 应该用什么精度？

| 格式 | 指数位 | 尾数位 | 最大值 | 特点 |
|---|---|---|---|---|
| fp16 | 5 | 10 | 65504 | 精度较高，范围小，需要 loss scaling |
| bf16 | 8 | 7 | ≈3.4e38 | 范围和 fp32 一样，精度低 |

归约类运算（softmax 分母、logsumexp、LayerNorm 的方差、loss 求和）要在 fp32 里累加，再 cast 回去。RL 里算 logprob 前一般先 `logits.float()`，否则同一个策略算出的 ratio 都不等于 1。

## mask 用 -inf 还是 -1e9？

- `-inf`：语义最干净，被 mask 的位置权重严格为 0。坑在于整行都被 mask 时 softmax 得到 `nan`，并会通过反向传播污染整个 batch。
- `-1e9`：整行 mask 时退化成均匀分布，不会 `nan`。但 fp16 最大只有 65504，`-1e9` 会溢出成 `-inf`。
- 稳妥写法：`torch.finfo(dtype).min`，并保证每个 query 至少有一个可见 key（或者把全 mask 行的输出置零）。

## causal mask 怎么构造？Tq ≠ Tk 时怎么对齐？

`Tq == Tk` 时就是下三角：`mask[i, j] = j <= i`。

带 KV cache 解码时 query 是最后 `Tq` 个位置，要**右下角对齐**：`j <= i + (Tk - Tq)`。`Tq=1` 时能看到全部 key。

注意 `F.scaled_dot_product_attention(is_causal=True)` 用的是左上角对齐，KV cache 场景要自己传 mask（或者 `Tq=1` 时直接不加 mask）。

## 从 logits 取 token logprob：gather 哪个维度？为什么 logits 和 labels 要错开一位？

```python
logp = torch.log_softmax(logits.float(), -1)               # (B, T, V)
tok_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # 在 vocab 维 gather
```

自回归模型第 $t$ 个位置的输出预测第 $t+1$ 个 token。所以 response 的 logprob 是 `logits[:, -R-1:-1]` 对齐 `input_ids[:, -R:]`。最后一个位置的 logits 预测的是序列之后的 token，训练时用不到。

动手：`exercises/model/logprob_entropy`

## vocab 很大时，算 logprob 为什么要分块或者用融合 kernel？

完整的 `(B, T, V)` fp32 log 概率张量非常大。比如 8 条 8192 token 的序列、vocab 152064：$8\times8192\times152064\times4\text{B}\approx 40\text{GB}$。

做法：只取被选中 token 的 logit 和整行 logsumexp（不物化整个 log_softmax）；沿序列分块计算；或者用融合的 cross-entropy kernel（verl 在可用时调用 flash-attn 的 `cross_entropy_loss`）。

## KV cache 的形状是什么？每个 token 占多少显存？

每层 K、V 各一份，形状 `(B, n_kv_heads, T, head_dim)`。每个 token 的字节数：

$$2 \times L \times n_{kv} \times d_{head} \times \text{bytes}$$

Llama-2-7B（MHA，$L=32$，$n_{kv}=32$，$d_{head}=128$，bf16）：$2\times32\times32\times128\times2=512\text{KB}$/token，4K 上下文约 2GB/条。GQA 把 $n_{kv}$ 降到 8 就省 4 倍。

动手：`exercises/model/mha_gqa_kvcache`

## RoPE 的 rotate_half 和 interleaved 实现有什么区别？能混用吗？

- interleaved：相邻两维 $(x_{2i},x_{2i+1})$ 成对旋转（原论文、Meta 原版 Llama 的复数写法）。
- rotate_half：前一半和后一半 $(x_i,x_{i+d/2})$ 成对旋转（HF transformers）。

两者相当于对 q、k 的维度做了同一个置换，数学上等价，但**同一份权重不能换实现用**。HF 转换 Meta 权重时会重排 `wq`/`wk` 的行。另外 RoPE 只加在 q、k 上，KV cache 里存的是旋转后的 k。

## 注意力为什么要除以 √d？

$q,k$ 的分量独立、方差为 1 时，$q\cdot k$ 的方差是 $d$。不缩放的话 $d$ 越大，softmax 的输入越大，输出越接近 one-hot，梯度接近 0。除以 $\sqrt d$ 把方差拉回 1。

## LayerNorm 的方差是有偏还是无偏？GRPO 的 std 呢？

- LayerNorm 用**有偏**方差（除以 $D$）。`np.var` 默认有偏；`torch.var` 默认无偏，手写时要传 `unbiased=False` / `correction=0`。
- verl 的 GRPO 用 `torch.std`，默认**无偏**（除以 $n-1$）。numpy 复现时要写 `np.std(x, ddof=1)`。

同一个“std”在不同库里默认值不同，这是对拍时最常见的数值不一致来源。

## PPO 的 ratio 为什么写成 exp(logp − old_logp)？为什么还要 clamp？

logprob 本来就在 log 空间，相减再 exp 比两个很小的概率相除稳定得多。verl 在 exp 之前把差值 clamp 到 $[-20, 20]$，防止 off-policy 或数值异常的 token 让 ratio 变成 `inf`，进而产生 `nan` 梯度。k3 KL 估计量也有类似的 clamp。

## token-mean 的 loss 在多卡数据并行时怎么归一化？

如果每张卡各自除以本卡的 token 数，再由 DDP/FSDP 对梯度取平均，得到的是“各卡均值的均值”，token 多的卡被低估，结果和卡数有关。

verl 的 `agg_loss` 用**全局** token 数做分母，再乘 `dp_size` 抵消梯度平均：每卡 loss $=\frac{\sum_{\text{本卡}}\ell}{N_{\text{global}}}\cdot \text{dp}$，平均后正好是 $\frac{\sum_{\text{全局}}\ell}{N_{\text{global}}}$。梯度累积（micro batch）也是同样的道理。

## 交叉熵里的 padding 怎么处理？

两种等价写法：label 设成 `-100`（`F.cross_entropy` 默认的 `ignore_index`），或者显式乘 mask 再除以有效 token 数。

常见 bug：先 `.mean()` 再乘 mask；或者分母用了 `B*T` 而不是有效 token 数，导致 padding 越多 loss 越小。

## 数值稳定的 log σ(x) 怎么写？

$\log\sigma(x) = -\log(1+e^{-x}) = -\text{softplus}(-x)$。numpy 用 `-np.logaddexp(0, -x)`，torch 用 `F.logsigmoid(x)`。

直接写 `np.log(1/(1+np.exp(-x)))` 时，$x=-1000$ 会让 `exp(1000)` 溢出，结果变成 `log(0) = -inf`，而真实值是 $-1000$。DPO、Bradley-Terry 奖励模型的 loss 都要用稳定写法。

## online softmax 的更新公式是什么？初值怎么处理？

维护行最大值 $m$、分母 $\ell$、输出累加 $O$。新块分数为 $S$：

$$m'=\max(m,\max S),\ \ \ell'=e^{m-m'}\ell+\textstyle\sum e^{S-m'},\ \ O'=e^{m-m'}O+e^{S-m'}V$$

最后 $O/\ell$。初值 $m=-\infty,\ \ell=0,\ O=0$；第一次更新时 $e^{-\infty}=0$。要防的是整块被 mask 时 $m'=-\infty$，出现 `exp(-inf - (-inf)) = nan`，一般把非有限的 $m'$ 替换成 0 再算指数。

动手：`exercises/model/flash_attention`

## 怎么用有限差分检查手写的反向传播？

用中心差分 $\frac{f(x+h)-f(x-h)}{2h}$，误差是 $O(h^2)$（前向差分是 $O(h)$）。在 float64 下取 $h\approx10^{-5}\sim10^{-6}$，对每个元素扰动一次，和解析梯度比相对误差（$10^{-6}$ 以内算通过）。

技巧：loss 取 $\sum y\odot G$（$G$ 随机），这样 `dy = G`，能一次检查任意上游梯度。float32 下 $h$ 太小会被舍入误差淹没。

动手：`exercises/model/layernorm_rmsnorm`

## Adam 的 eps 加在哪里？为什么需要 bias correction？

PyTorch 写法：$\theta \leftarrow \theta - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$，$\epsilon$ 在根号外面。

$m,v$ 初始化为 0，前几步严重偏小（第 1 步 $m=0.1g$，$v=0.001g^2$）。除以 $1-\beta^t$ 之后第 1 步 $\hat m=g,\ \hat v=g^2$，步长约为 $\eta$，不会因为 $v$ 偏小而在开头迈出过大或过小的步子。

动手：`exercises/model/adam`

## top-k 采样遇到并列怎么办？top-p 至少保留几个？

top-k 常见实现是“保留概率 ≥ 第 k 大值的 token”，边界并列会全部保留，实际可能多于 k 个；用 `topk` 取下标则会任意截断并列项。top-p 要保留“让累计概率第一次达到 p 的那个 token”，并且至少保留 1 个，否则 p 很小时会得到空分布。

RL rollout 一般关掉 top-k/top-p（`top_p=1, top_k=-1`），否则采样分布和训练时算 logprob 的分布不一致。

动手：`exercises/model/sampling`
