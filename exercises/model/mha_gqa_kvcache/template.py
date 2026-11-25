import numpy as np


def repeat_kv(x, n_rep):
    """(B, n_kv, T, hd) -> (B, n_kv * n_rep, T, hd)，新 head h 对应原 head h // n_rep。"""
    raise NotImplementedError


def gqa_attention(x, params, n_heads, n_kv_heads, causal=True):
    """GQA 前向。

    Args:
        x: (B, T, d_model)
        params: {"wq": (d_model, n_heads*hd), "wk": (d_model, n_kv*hd),
                 "wv": (d_model, n_kv*hd), "wo": (n_heads*hd, d_model)}
    Returns:
        (B, T, d_model)
    """
    raise NotImplementedError


class KVCache:
    """按层保存的 KV cache；k/v 形状 (B, n_kv_heads, T_cached, hd)，空时为 None。"""

    def __init__(self):
        raise NotImplementedError

    def append(self, k_new, v_new):
        """k_new/v_new: (B, n_kv_heads, t, hd)。沿时间维拼接，返回拼接后的 (k, v)。"""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def gqa_decode_step(x_new, params, n_heads, n_kv_heads, cache):
    """增量解码：x_new (B, t, d_model)，返回 (B, t, d_model)，并更新 cache。"""
    raise NotImplementedError
