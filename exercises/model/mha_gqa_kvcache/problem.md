---
title: GQA 多头注意力 + KV Cache 增量解码
difficulty: 中
tags: Transformer, GQA, KV cache, 推理
order: 16
source: Ainslie et al. 2023 GQA；Llama 的 repeat_kv
---

实现 grouped-query attention（GQA）的前向，以及带 KV cache 的增量解码。

输入、权重和输出都是 `torch.Tensor`（测试用 float64），输出 dtype 与 `x` 相同。权重就是普通 tensor，放在 dict 里传进来，不需要写 `nn.Module`。

**参数约定**（行向量右乘权重，`x @ wq`）：`params = {"wq", "wk", "wv", "wo"}`

| 权重 | 形状 |
|---|---|
| `wq` | `(d_model, n_heads * hd)` |
| `wk`, `wv` | `(d_model, n_kv_heads * hd)` |
| `wo` | `(n_heads * hd, d_model)` |

`hd = wq.shape[1] // n_heads`，`n_heads` 一定能被 `n_kv_heads` 整除。投影后按 `(B, T, n, hd) → (B, n, T, hd)` 拆 head。**第 `h` 个 query head 使用第 `h // (n_heads // n_kv_heads)` 个 KV head**（Llama 的 `repeat_kv` 约定）。

```python
repeat_kv(x, n_rep)                    # (B, n_kv, T, hd) -> (B, n_kv*n_rep, T, hd)
gqa_attention(x, params, n_heads, n_kv_heads, causal=True)   # x: (B, T, d_model) -> (B, T, d_model)

class KVCache:
    k, v                               # 形状 (B, n_kv_heads, T_cached, hd) 的 tensor，空 cache 时为 None
    def append(self, k_new, v_new)     # 沿时间维拼接，返回拼接后的 (k, v)
    def __len__(self)                  # 已缓存的 token 数（python int）

gqa_decode_step(x_new, params, n_heads, n_kv_heads, cache)  # x_new: (B, t, d_model)，t>=1
```

`gqa_decode_step` 只对新 token 计算 q/k/v，把新的 k/v 追加进 cache，再让新 query 对**全部**缓存的 key 做 causal 注意力（右下角对齐），返回 `(B, t, d_model)`。

注意力核心自己写（`torch.softmax`、`torch.repeat_interleave` 可以用），**不要直接调用** `F.scaled_dot_product_attention`（尤其是它的 `enable_gqa=True`，测试拿它当对照）或 `nn.MultiheadAttention`。

**核心性质（测试会检查）**：逐 token 解码的输出 = 一次性 causal 前向的输出。

<!-- solution -->

### 讲解

**GQA 的动机**：解码阶段每步都要把整个 KV cache 从 HBM 读一遍，是访存瓶颈。KV cache 大小正比于 KV head 数：

$$\text{KV bytes} = 2 \cdot L \cdot n_{kv} \cdot hd \cdot T \cdot \text{bytes}$$

MHA（$n_{kv}=n_h$）→ GQA（如 Llama-2-70B 是 64 个 query head 对 8 个 KV head）→ MQA（$n_{kv}=1$），cache 依次缩小 8 倍、64 倍。质量上 GQA 接近 MHA，MQA 掉点更明显。

**实现要点**

- cache 里存的是 **n_kv 个 head**，不是 repeat 之后的。repeat 只在算注意力时临时做（真正的 kernel 连复制都不做，直接按 `h // n_rep` 索引）。
- `repeat_kv` 要的是 `[h0,h0,h1,h1,...]`（`torch.repeat_interleave(x, n_rep, dim=1)` 或下面的 expand + reshape），不是 `x.repeat(1, n_rep, 1, 1)`：后者得到 `[h0,h1,h0,h1,...]`，head 对应关系就错了。
- 解码时新 query 的位置是 `T_cached + i`，causal mask 要右下角对齐。
- 预填充（prefill）可以一次喂多个 token，`gqa_decode_step` 支持 `t > 1` 就同时覆盖 prefill 和 decode。

**复杂度**：每步 decode 的注意力是 $O(n_h \cdot T \cdot hd)$ 计算，但要读 $O(n_{kv} \cdot T \cdot hd)$ 的 cache。没有 cache 时每步都要重算整个前缀，总代价从 $O(T^2)$ 变成 $O(T^3)$。

> [!Q]
> KV cache 为什么只存 K、V，不存 Q？
> 解码第 $t$ 步只需要新 token 的 query 去和历史所有 key 做点积；历史 token 的 query 以后再也用不到。K、V 会被之后的每一步重复使用，所以缓存它们。

> [!Q]
> 怎么把一个 MHA checkpoint 转成 GQA？
> GQA 论文的做法是把同一组内的 K/V head 权重取平均（mean pooling）得到新的 KV head，再用少量（约 5% 预训练算力）数据做 uptraining。

```python
# torch 版 repeat_kv（Llama）
def repeat_kv(x, n_rep):
    b, n_kv, t, hd = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)
```
