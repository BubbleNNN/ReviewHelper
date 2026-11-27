import math

import torch

_C = math.sqrt(2.0 / math.pi)


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(_C * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    u = _C * (x + 0.044715 * x ** 3)
    t = torch.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * _C * (1.0 + 3 * 0.044715 * x ** 2)


def _sum_all_reduce(tensors):
    return torch.stack(tensors).sum(dim=0)


def shard_weights(W1, b1, W2, tp):
    cols = torch.tensor_split(torch.arange(W1.shape[1], device=W1.device), tp)
    return [{"W1": W1[:, c].clone(), "b1": b1[c].clone(), "W2": W2[c, :].clone()} for c in cols]


def tp_mlp_forward(x, shards, b2, all_reduce=None):
    all_reduce = all_reduce or _sum_all_reduce
    zs, partials = [], []
    for s in shards:                       # 每个 rank 独立计算，无通信
        z = x @ s["W1"] + s["b1"]          # column parallel
        zs.append(z)
        partials.append(gelu(z) @ s["W2"])  # row parallel → 部分和
    y = all_reduce(partials) + b2           # g: 前向 all-reduce，b2 只加一次
    return y, {"z": zs}


def tp_mlp_backward(dy, x, shards, cache, all_reduce=None):
    all_reduce = all_reduce or _sum_all_reduce
    grads = {"db2": dy.sum(dim=0), "dW1": [], "db1": [], "dW2": []}
    dx_parts = []
    for s, z in zip(shards, cache["z"]):   # g 的反向是 identity：每个 rank 都拿到完整 dy
        a = gelu(z)
        grads["dW2"].append(a.T @ dy)
        dz = (dy @ s["W2"].T) * gelu_grad(z)
        grads["dW1"].append(x.T @ dz)
        grads["db1"].append(dz.sum(dim=0))
        dx_parts.append(dz @ s["W1"].T)
    grads["dx"] = all_reduce(dx_parts)      # f: 反向 all-reduce
    return grads
