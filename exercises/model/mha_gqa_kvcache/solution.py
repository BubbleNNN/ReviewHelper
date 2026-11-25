import numpy as np


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=1)  # [h0,h0,h1,h1,...]：新 head h 对应 h // n_rep


def _split_heads(x, n, hd):
    B, T, _ = x.shape
    return x.reshape(B, T, n, hd).transpose(0, 2, 1, 3)


def _merge_heads(x):
    B, n, T, hd = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, n * hd)


def _attend(q, k, v, causal):
    hd = q.shape[-1]
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    Tq, Tk = s.shape[-2:]
    if causal:
        allowed = np.arange(Tk)[None, :] <= np.arange(Tq)[:, None] + (Tk - Tq)
        s = np.where(allowed, s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v


def _qkv(x, params, n_heads, n_kv_heads):
    hd = params["wq"].shape[1] // n_heads
    q = _split_heads(x @ params["wq"], n_heads, hd)
    k = _split_heads(x @ params["wk"], n_kv_heads, hd)
    v = _split_heads(x @ params["wv"], n_kv_heads, hd)
    return q, k, v


def gqa_attention(x, params, n_heads, n_kv_heads, causal=True):
    x = np.asarray(x, dtype=np.float64)
    q, k, v = _qkv(x, params, n_heads, n_kv_heads)
    rep = n_heads // n_kv_heads
    o = _attend(q, repeat_kv(k, rep), repeat_kv(v, rep), causal)
    return _merge_heads(o) @ params["wo"]


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k_new, v_new):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = np.concatenate([self.k, k_new], axis=2)
            self.v = np.concatenate([self.v, v_new], axis=2)
        return self.k, self.v

    def __len__(self):
        return 0 if self.k is None else self.k.shape[2]


def gqa_decode_step(x_new, params, n_heads, n_kv_heads, cache):
    x_new = np.asarray(x_new, dtype=np.float64)
    q, k, v = _qkv(x_new, params, n_heads, n_kv_heads)
    k_all, v_all = cache.append(k, v)
    rep = n_heads // n_kv_heads
    o = _attend(q, repeat_kv(k_all, rep), repeat_kv(v_all, rep), causal=True)
    return _merge_heads(o) @ params["wo"]
