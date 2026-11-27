import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _setup(n=5, d=6, h=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=F64)
    W1 = torch.randn(d, h, generator=g, dtype=F64) / math.sqrt(d)
    b1 = torch.randn(h, generator=g, dtype=F64) * 0.1
    W2 = torch.randn(h, d, generator=g, dtype=F64) / math.sqrt(h)
    b2 = torch.randn(d, generator=g, dtype=F64) * 0.1
    return x, W1, b1, W2, b2


def _randn(seed, *shape):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=F64)


def _full(x, W1, b1, W2, b2):
    # 不切分的完整 MLP；用 torch 内置的 tanh 近似 GeLU 当 oracle
    return F.gelu(x @ W1 + b1, approximate="tanh") @ W2 + b2


def _autograd_grads(x, W1, b1, W2, b2, G):
    """在完整 MLP 上用 autograd 求标量损失 L = sum(Y * G) 的梯度（于是 dL/dY = G）。"""
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, W1, b1, W2, b2)]
    loss = (_full(*leaves) * G).sum()
    return dict(zip(("dx", "dW1", "db1", "dW2", "db2"), torch.autograd.grad(loss, leaves)))


def _loss(x, W1, b1, W2, b2, G):
    return float((_full(x, W1, b1, W2, b2) * G).sum())


def _num_grad(f, a, eps=1e-6):
    """中心差分，逐个扰动 a 的元素（a 需连续，原地改完会还原）。"""
    g = torch.zeros_like(a)
    flat, gflat = a.view(-1), g.view(-1)
    for i in range(a.numel()):
        old = flat[i].item()
        flat[i] = old + eps
        fp = f()
        flat[i] = old - eps
        fm = f()
        flat[i] = old
        gflat[i] = (fp - fm) / (2 * eps)
    return g


class _Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self, tensors):
        self.calls += 1
        return torch.stack(tensors).sum(dim=0)


def test_shard_shapes(m):
    x, W1, b1, W2, b2 = _setup(h=10)
    shards = m.shard_weights(W1, b1, W2, 4)
    assert len(shards) == 4
    assert [s["W1"].shape[1] for s in shards] == [3, 3, 2, 2]  # 和 torch.tensor_split 一致
    _close(torch.cat([s["W1"] for s in shards], dim=1), W1, atol=0)
    _close(torch.cat([s["b1"] for s in shards]), b1, atol=0)
    _close(torch.cat([s["W2"] for s in shards], dim=0), W2, atol=0)


def test_forward_matches_full(m):
    for tp in (1, 2, 3, 4):
        x, W1, b1, W2, b2 = _setup(seed=tp)
        y, _ = m.tp_mlp_forward(x, m.shard_weights(W1, b1, W2, tp), b2)
        _close(y, _full(x, W1, b1, W2, b2), atol=1e-10)


def test_forward_one_allreduce_and_bias_once(m):
    x, W1, b1, W2, b2 = _setup(seed=7)
    cnt = _Counter()
    y, _ = m.tp_mlp_forward(x, m.shard_weights(W1, b1, W2, 4), b2, all_reduce=cnt)
    assert cnt.calls == 1, "前向只允许一次 all-reduce"
    _close(y, _full(x, W1, b1, W2, b2), atol=1e-10)


def test_backward_dx_matches_numeric(m):
    x, W1, b1, W2, b2 = _setup(n=3, d=4, h=6, seed=11)
    G = _randn(12, 3, 4)
    shards = m.shard_weights(W1, b1, W2, 3)
    y, cache = m.tp_mlp_forward(x, shards, b2)
    cnt = _Counter()
    g = m.tp_mlp_backward(G, x, shards, cache, all_reduce=cnt)
    assert cnt.calls == 1, "反向只允许一次 all-reduce（对 dx 求和）"
    _close(g["dx"], _autograd_grads(x, W1, b1, W2, b2, G)["dx"], atol=1e-10)
    num_dx = _num_grad(lambda: _loss(x, W1, b1, W2, b2, G), x)
    _close(g["dx"], num_dx, atol=1e-6)


def test_backward_weight_grads(m):
    x, W1, b1, W2, b2 = _setup(n=3, d=4, h=5, seed=21)
    G = _randn(22, 3, 4)
    shards = m.shard_weights(W1, b1, W2, 2)
    _, cache = m.tp_mlp_forward(x, shards, b2)
    g = m.tp_mlp_backward(G, x, shards, cache)
    ref = _autograd_grads(x, W1, b1, W2, b2, G)
    _close(torch.cat(g["dW1"], dim=1), ref["dW1"], atol=1e-10)
    _close(torch.cat(g["db1"]), ref["db1"], atol=1e-10)
    _close(torch.cat(g["dW2"], dim=0), ref["dW2"], atol=1e-10)
    _close(g["db2"], ref["db2"], atol=1e-10)


def test_uneven_split_backward(m):
    x, W1, b1, W2, b2 = _setup(n=2, d=3, h=7, seed=31)
    G = torch.ones(2, 3, dtype=F64)
    ref = _autograd_grads(x, W1, b1, W2, b2, G)
    for tp in (2, 4):
        shards = m.shard_weights(W1, b1, W2, tp)
        _, cache = m.tp_mlp_forward(x, shards, b2)
        g = m.tp_mlp_backward(G, x, shards, cache)
        _close(g["dx"], ref["dx"], atol=1e-10)
        _close(torch.cat(g["dW1"], dim=1), ref["dW1"], atol=1e-10)
        _close(torch.cat(g["dW2"], dim=0), ref["dW2"], atol=1e-10)
