import numpy as np

_C = np.sqrt(2.0 / np.pi)


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(_C * (x + 0.044715 * x ** 3)))


def _setup(n=5, d=6, h=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    W1 = rng.normal(size=(d, h)) / np.sqrt(d)
    b1 = rng.normal(size=h) * 0.1
    W2 = rng.normal(size=(h, d)) / np.sqrt(h)
    b2 = rng.normal(size=d) * 0.1
    return x, W1, b1, W2, b2


def _full(x, W1, b1, W2, b2):
    return _gelu(x @ W1 + b1) @ W2 + b2


def _loss(x, W1, b1, W2, b2, G):
    # 标量损失 L = sum(Y * G)，于是 dL/dY = G
    return float(np.sum(_full(x, W1, b1, W2, b2) * G))


def _num_grad(f, a, eps=1e-6):
    g = np.zeros_like(a)
    it = np.nditer(a, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = a[i]
        a[i] = old + eps
        fp = f()
        a[i] = old - eps
        fm = f()
        a[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


class _Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self, arrays):
        self.calls += 1
        return np.sum(arrays, axis=0)


def test_shard_shapes(m):
    x, W1, b1, W2, b2 = _setup(h=10)
    shards = m.shard_weights(W1, b1, W2, 4)
    assert len(shards) == 4
    assert [s["W1"].shape[1] for s in shards] == [3, 3, 2, 2]  # np.array_split
    np.testing.assert_allclose(np.concatenate([s["W1"] for s in shards], axis=1), W1)
    np.testing.assert_allclose(np.concatenate([s["b1"] for s in shards]), b1)
    np.testing.assert_allclose(np.concatenate([s["W2"] for s in shards], axis=0), W2)


def test_forward_matches_full(m):
    for tp in (1, 2, 3, 4):
        x, W1, b1, W2, b2 = _setup(seed=tp)
        y, _ = m.tp_mlp_forward(x, m.shard_weights(W1, b1, W2, tp), b2)
        np.testing.assert_allclose(y, _full(x, W1, b1, W2, b2), atol=1e-10)


def test_forward_one_allreduce_and_bias_once(m):
    x, W1, b1, W2, b2 = _setup(seed=7)
    cnt = _Counter()
    y, _ = m.tp_mlp_forward(x, m.shard_weights(W1, b1, W2, 4), b2, all_reduce=cnt)
    assert cnt.calls == 1, "前向只允许一次 all-reduce"
    np.testing.assert_allclose(y, _full(x, W1, b1, W2, b2), atol=1e-10)


def test_backward_dx_matches_numeric(m):
    x, W1, b1, W2, b2 = _setup(n=3, d=4, h=6, seed=11)
    G = np.random.default_rng(12).normal(size=(3, 4))
    shards = m.shard_weights(W1, b1, W2, 3)
    y, cache = m.tp_mlp_forward(x, shards, b2)
    cnt = _Counter()
    g = m.tp_mlp_backward(G, x, shards, cache, all_reduce=cnt)
    assert cnt.calls == 1, "反向只允许一次 all-reduce（对 dx 求和）"
    num_dx = _num_grad(lambda: _loss(x, W1, b1, W2, b2, G), x)
    np.testing.assert_allclose(g["dx"], num_dx, atol=1e-6)


def test_backward_weight_grads(m):
    x, W1, b1, W2, b2 = _setup(n=3, d=4, h=5, seed=21)
    G = np.random.default_rng(22).normal(size=(3, 4))
    shards = m.shard_weights(W1, b1, W2, 2)
    _, cache = m.tp_mlp_forward(x, shards, b2)
    g = m.tp_mlp_backward(G, x, shards, cache)
    f = lambda: _loss(x, W1, b1, W2, b2, G)  # noqa: E731
    np.testing.assert_allclose(np.concatenate(g["dW1"], axis=1), _num_grad(f, W1), atol=1e-6)
    np.testing.assert_allclose(np.concatenate(g["db1"]), _num_grad(f, b1), atol=1e-6)
    np.testing.assert_allclose(np.concatenate(g["dW2"], axis=0), _num_grad(f, W2), atol=1e-6)
    np.testing.assert_allclose(g["db2"], _num_grad(f, b2), atol=1e-6)


def test_uneven_split_backward(m):
    x, W1, b1, W2, b2 = _setup(n=2, d=3, h=7, seed=31)
    G = np.ones((2, 3))
    for tp in (2, 4):
        shards = m.shard_weights(W1, b1, W2, tp)
        _, cache = m.tp_mlp_forward(x, shards, b2)
        g = m.tp_mlp_backward(G, x, shards, cache)
        ref = _num_grad(lambda: _loss(x, W1, b1, W2, b2, G), x)
        np.testing.assert_allclose(g["dx"], ref, atol=1e-6)
