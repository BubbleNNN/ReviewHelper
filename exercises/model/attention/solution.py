import numpy as np


def scaled_dot_product_attention(q, k, v, causal=False, key_padding_mask=None):
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)  # (B, H, Tq, Tk)
    Tq, Tk = scores.shape[-2:]

    allowed = np.ones((Tq, Tk), dtype=bool)
    if causal:
        allowed = np.arange(Tk)[None, :] <= np.arange(Tq)[:, None] + (Tk - Tq)
    allowed = np.broadcast_to(allowed, scores.shape)
    if key_padding_mask is not None:
        kpm = np.asarray(key_padding_mask, dtype=bool)[:, None, None, :]
        allowed = allowed & kpm

    scores = np.where(allowed, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    attn = w / w.sum(axis=-1, keepdims=True)
    return attn @ v, attn
