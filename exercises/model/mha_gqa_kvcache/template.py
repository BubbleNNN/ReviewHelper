import math

import torch


def repeat_kv(x, n_rep):
    """(B, n_kv, T, hd) -> (B, n_kv * n_rep, T, hd)，新 head h 对应原 head h // n_rep。

    x 是 float tensor，返回 dtype 相同；n_rep=1 时原样返回。
    """
    raise NotImplementedError


def gqa_attention(x, params, n_heads, n_kv_heads, causal=True):
    """GQA 前向。不要调用 F.scaled_dot_product_attention。

    Args:
        x: (B, T, d_model) float tensor
        params: dict，值都是与 x 同 dtype 的 tensor（行向量右乘）：
                {"wq": (d_model, n_heads*hd), "wk": (d_model, n_kv*hd),
                 "wv": (d_model, n_kv*hd), "wo": (n_heads*hd, d_model)}
        n_heads, n_kv_heads: python int，n_heads 能被 n_kv_heads 整除
        causal: True 时用 causal mask
    Returns:
        (B, T, d_model) tensor，dtype 与 x 相同
    """
    raise NotImplementedError


class KVCache:
    """按层保存的 KV cache；k/v 形状 (B, n_kv_heads, T_cached, hd) 的 tensor，空时为 None。"""

    def __init__(self):
        raise NotImplementedError

    def append(self, k_new, v_new):
        """k_new/v_new: (B, n_kv_heads, t, hd) tensor。沿时间维（dim=2）拼接，返回拼接后的 (k, v)。"""
        raise NotImplementedError

    def __len__(self):
        """已缓存的 token 数（python int），空 cache 为 0。"""
        raise NotImplementedError


def gqa_decode_step(x_new, params, n_heads, n_kv_heads, cache):
    """增量解码：只对新 token 算 q/k/v，把新 k/v 追加进 cache，再对全部缓存做右下角对齐的 causal 注意力。

    Args:
        x_new: (B, t, d_model) float tensor，t>=1（t>1 即 prefill / 分块 prefill）
        cache: KVCache，会被原地更新
    Returns:
        (B, t, d_model) tensor，dtype 与 x_new 相同
    """
    raise NotImplementedError
