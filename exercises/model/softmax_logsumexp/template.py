import numpy as np


def logsumexp(x, axis=-1, keepdims=False):
    """数值稳定的 log Σ exp(x)。

    Args:
        x: 任意形状数组，可以包含 -inf
        axis: 归约的维度
        keepdims: 是否保留被归约的维度（长度为 1）
    Returns:
        归约后的数组；整行 -inf 时结果为 -inf
    """
    raise NotImplementedError


def softmax(x, axis=-1):
    """数值稳定的 softmax，-inf 位置输出 0。形状与 x 相同。"""
    raise NotImplementedError


def log_softmax(x, axis=-1):
    """x - logsumexp(x)，-inf 位置输出 -inf。形状与 x 相同。"""
    raise NotImplementedError
