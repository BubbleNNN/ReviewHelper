import numpy as np


def rope_cos_sin(positions, dim, base=10000.0):
    """返回 cos, sin 表，形状各为 (len(positions), dim)。

    θ_i = base^(-2i/dim)，i = 0..dim/2-1；表 = concat([pos*θ, pos*θ]) 再取 cos/sin。
    """
    raise NotImplementedError


def rotate_half(x):
    """(..., dim)：前后两半 x1, x2 -> concat([-x2, x1])。"""
    raise NotImplementedError


def apply_rope(x, positions, base=10000.0):
    """x: (..., T, dim)，positions: (T,) -> 同形状。"""
    raise NotImplementedError
