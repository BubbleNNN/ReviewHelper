import numpy as np


def rope_cos_sin(positions, dim, base=10000.0):
    positions = np.asarray(positions, dtype=np.float64)
    inv_freq = base ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)  # (dim/2,)
    ang = positions[:, None] * inv_freq[None, :]
    emb = np.concatenate([ang, ang], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(x, positions, base=10000.0):
    x = np.asarray(x, dtype=np.float64)
    cos, sin = rope_cos_sin(positions, x.shape[-1], base)
    return x * cos + rotate_half(x) * sin
