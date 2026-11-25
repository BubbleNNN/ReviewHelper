import numpy as np


def _make(m, out_f=6, in_f=5, r=2, alpha=8.0, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(out_f, in_f))
    return W, m.LoRALinear(W, r, alpha, np.random.default_rng(seed + 100))


def test_init_matches_base(m):
    W, layer = _make(m)
    assert layer.A.shape == (2, 5) and layer.B.shape == (6, 2)
    assert np.all(layer.B == 0) and np.any(layer.A != 0)
    np.testing.assert_allclose(layer.scaling, 4.0)
    assert layer.merged is False
    x = np.random.default_rng(1).normal(size=(3, 5))
    np.testing.assert_allclose(layer.forward(x), x @ W.T, atol=1e-12)


def test_forward_formula(m):
    W, layer = _make(m)
    rng = np.random.default_rng(2)
    layer.A = rng.normal(size=(2, 5))
    layer.B = rng.normal(size=(6, 2))
    x = rng.normal(size=(4, 5))
    expected = x @ W.T + 4.0 * x @ layer.A.T @ layer.B.T
    np.testing.assert_allclose(layer.forward(x), expected, atol=1e-12)
    x3 = rng.normal(size=(2, 3, 5))  # 任意前导维度
    np.testing.assert_allclose(layer.forward(x3)[1, 2], layer.forward(x3[1, 2:3])[0], atol=1e-12)


def test_merge_equivalence_and_idempotent(m):
    W, layer = _make(m)
    rng = np.random.default_rng(3)
    layer.A = rng.normal(size=(2, 5))
    layer.B = rng.normal(size=(6, 2))
    x = rng.normal(size=(4, 5))
    before = layer.forward(x)
    layer.merge()
    assert layer.merged is True
    np.testing.assert_allclose(layer.forward(x), before, atol=1e-10)
    np.testing.assert_allclose(layer.W, W + 4.0 * layer.B @ layer.A, atol=1e-12)
    layer.merge()  # 再合并一次不应叠加
    np.testing.assert_allclose(layer.forward(x), before, atol=1e-10)


def test_unmerge_restores(m):
    W, layer = _make(m)
    rng = np.random.default_rng(4)
    layer.unmerge()  # 未合并时什么也不做
    np.testing.assert_allclose(layer.W, W)
    layer.A = rng.normal(size=(2, 5))
    layer.B = rng.normal(size=(6, 2))
    layer.merge()
    layer.unmerge()
    assert layer.merged is False
    np.testing.assert_allclose(layer.W, W, atol=1e-12)


def test_does_not_mutate_input(m):
    W, layer = _make(m)
    W_copy = W.copy()
    layer.B = np.ones_like(layer.B)
    layer.merge()
    np.testing.assert_array_equal(W, W_copy)


def test_param_count(m):
    _, layer = _make(m, out_f=512, in_f=256, r=4, alpha=8.0)
    assert layer.num_trainable_params() == 4 * (512 + 256)  # r·(out + in)，远小于 512·256
