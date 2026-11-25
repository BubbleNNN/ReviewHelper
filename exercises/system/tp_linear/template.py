import numpy as np

_C = np.sqrt(2.0 / np.pi)


def gelu(x):
    """GeLU（tanh 近似），已给出。"""
    return 0.5 * x * (1.0 + np.tanh(_C * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    """GeLU 对输入的导数，已给出。"""
    u = _C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * _C * (1.0 + 3 * 0.044715 * x ** 2)


def _sum_all_reduce(arrays):
    return np.sum(arrays, axis=0)


def shard_weights(W1, b1, W2, tp):
    """W1/b1 按列（隐藏维 h）切，W2 按行切。用 np.array_split。

    Returns: 长度 tp 的 list，每个元素 {"W1": (d, h_r), "b1": (h_r,), "W2": (h_r, d)}
    """
    raise NotImplementedError


def tp_mlp_forward(x, shards, b2, all_reduce=None):
    """张量并行前向。只允许调用一次 all_reduce。

    Returns:
        y: (n, d)
        cache: 反向需要的中间量（自己定义）
    """
    all_reduce = all_reduce or _sum_all_reduce
    raise NotImplementedError


def tp_mlp_backward(dy, x, shards, cache, all_reduce=None):
    """张量并行反向。只允许调用一次 all_reduce（用于 dx）。

    Returns: {"dx", "db2", "dW1": [..], "db1": [..], "dW2": [..]}
    """
    all_reduce = all_reduce or _sum_all_reduce
    raise NotImplementedError
