import torch


def rope_cos_sin(positions, dim, base=10000.0):
    pos = positions.to(torch.float64)
    # arange 默认是 int64，除以 dim 会变成默认 dtype（float32），这里显式用 float64
    inv_freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim)  # (dim/2,)
    ang = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, positions, base=10000.0):
    cos, sin = rope_cos_sin(positions, x.shape[-1], base)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)  # (T, dim)，在前导维度上自动广播
    return x * cos + rotate_half(x) * sin
