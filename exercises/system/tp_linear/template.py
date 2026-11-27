import math

import torch

_C = math.sqrt(2.0 / math.pi)


def gelu(x):
    """GeLU（tanh 近似），已给出。x: 任意形状的 float tensor，返回同形状同 dtype。"""
    return 0.5 * x * (1.0 + torch.tanh(_C * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    """GeLU 对输入的导数（逐元素），已给出。x: 任意形状的 float tensor，返回同形状同 dtype。"""
    u = _C * (x + 0.044715 * x ** 3)
    t = torch.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * _C * (1.0 + 3 * 0.044715 * x ** 2)


def _sum_all_reduce(tensors):
    """默认的通信原语：list of 同形状 tensor -> 逐元素求和后的 tensor。"""
    return torch.stack(tensors).sum(dim=0)


def shard_weights(W1, b1, W2, tp):
    """W1/b1 按列（隐藏维 h）切，W2 按行切。用 torch.tensor_split 的切法（前 h % tp 份各多一个）。

    Args:
        W1: (d, h) float tensor；b1: (h,)；W2: (h, d)；tp: python int
    Returns:
        长度 tp 的 list，每个元素 {"W1": (d, h_r), "b1": (h_r,), "W2": (h_r, d)}，
        dtype 与输入相同
    """
    raise NotImplementedError


def tp_mlp_forward(x, shards, b2, all_reduce=None):
    """张量并行前向。只允许调用一次 all_reduce。

    Args:
        x: (n, d) float tensor；shards: shard_weights 的返回值；b2: (d,) float tensor
        all_reduce: list[tensor] -> tensor 的通信原语，默认逐元素求和
    Returns:
        y: (n, d) tensor，dtype 与 x 相同
        cache: 反向需要的中间量（自己定义）
    """
    all_reduce = all_reduce or _sum_all_reduce
    raise NotImplementedError


def tp_mlp_backward(dy, x, shards, cache, all_reduce=None):
    """张量并行反向（手写，不能用 autograd）。只允许调用一次 all_reduce（用于 dx）。

    Args:
        dy: (n, d) float tensor，dL/dY；x、shards 同前向；cache: 前向返回的 cache
    Returns:
        dict，值都是 tensor（dtype 与 x 相同）：
        {"dx": (n, d), "db2": (d,),
         "dW1": [每 rank 一个 (d, h_r)], "db1": [每 rank 一个 (h_r,)], "dW2": [每 rank 一个 (h_r, d)]}
    """
    all_reduce = all_reduce or _sum_all_reduce
    raise NotImplementedError
