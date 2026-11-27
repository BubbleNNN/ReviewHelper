import torch


def rope_cos_sin(positions, dim, base=10000.0):
    """返回 cos, sin 表。

    θ_i = base^(-2i/dim)，i = 0..dim/2-1；表 = concat([pos*θ, pos*θ]) 再取 cos/sin。

    Args:
        positions: (T,) long tensor，位置下标
        dim: python int，偶数
        base: python float
    Returns:
        cos, sin: 各 (T, dim) 的 float64 tensor（不管 positions 是什么 dtype，都用 float64 算），
                  device 与 positions 相同
    """
    raise NotImplementedError


def rotate_half(x):
    """x: (..., dim) tensor，前后两半 x1, x2 -> concat([-x2, x1])。返回同形状、同 dtype。"""
    raise NotImplementedError


def apply_rope(x, positions, base=10000.0):
    """对最后一维施加 RoPE（rotate_half 约定）。

    Args:
        x: (..., T, dim) float tensor，比如 (B, H, T, dim)
        positions: (T,) long tensor
    Returns:
        与 x 同形状、同 dtype 的 tensor
    """
    raise NotImplementedError
