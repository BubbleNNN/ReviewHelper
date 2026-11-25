import numpy as np


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """LayerNorm 前向。

    Args:
        x:     (N, D)
        gamma: (D,)
        beta:  (D,)
    Returns:
        y:     (N, D)
        cache: 反向需要的中间量（内容自定）
    """
    raise NotImplementedError


def layernorm_backward(dy, cache):
    """LayerNorm 反向。

    Args:
        dy:    (N, D) 上游梯度
        cache: layernorm_forward 返回的 cache
    Returns:
        dx:     (N, D)
        dgamma: (D,)
        dbeta:  (D,)
    """
    raise NotImplementedError


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm：x / sqrt(mean(x^2) + eps) * weight。x: (..., D)，weight: (D,)"""
    raise NotImplementedError
