import math

import numpy as np


def test_layernorm_hand_computed(m):
    # x=[1,2,3]：μ=2，σ²=2/3，x̂ = [-1, 0, 1] / sqrt(2/3)
    y, _ = m.layernorm_forward(np.array([[1.0, 2.0, 3.0]]), np.ones(3), np.zeros(3), eps=0.0)
    s = 1.0 / math.sqrt(2.0 / 3.0)
    np.testing.assert_allclose(y, [[-s, 0.0, s]], atol=1e-12)


def test_layernorm_normalizes_and_affine(m):
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 5.0, size=(6, 16))
    y, _ = m.layernorm_forward(x, np.ones(16), np.zeros(16), eps=1e-5)
    np.testing.assert_allclose(y.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose((y ** 2).mean(axis=1), 1.0, atol=1e-5)  # 有偏方差≈1
    gamma = rng.normal(size=16)
    beta = rng.normal(size=16)
    y2, _ = m.layernorm_forward(x, gamma, beta, eps=1e-5)
    np.testing.assert_allclose(y2, y * gamma + beta, atol=1e-10)


def test_rmsnorm_hand_computed(m):
    # x=[3,4]：mean(x²)=12.5，rms=sqrt(12.5)
    r = math.sqrt(12.5)
    y = m.rmsnorm(np.array([[3.0, 4.0]]), np.array([1.0, 1.0]), eps=0.0)
    np.testing.assert_allclose(y, [[3 / r, 4 / r]], atol=1e-12)
    y2 = m.rmsnorm(np.array([[3.0, 4.0]]), np.array([2.0, 0.5]), eps=0.0)
    np.testing.assert_allclose(y2, [[6 / r, 2 / r]], atol=1e-12)


def test_rmsnorm_properties(m):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 8))
    w = np.ones(8)
    y = m.rmsnorm(x, w, eps=0.0)
    assert y.shape == x.shape
    np.testing.assert_allclose(m.rmsnorm(7.5 * x, w, eps=0.0), y, atol=1e-12)  # 正缩放不变
    np.testing.assert_allclose((y ** 2).mean(axis=-1), 1.0, atol=1e-12)
    # 不减均值：整体平移会改变输出（和 LayerNorm 的区别）
    assert not np.allclose(m.rmsnorm(x + 3.0, w, eps=0.0), y)


def _numeric_grads(m, x, gamma, beta, G, eps, h=1e-6):
    def loss(x_, g_, b_):
        y, _ = m.layernorm_forward(x_, g_, b_, eps=eps)
        return float((y * G).sum())

    grads = []
    for arr_idx, arr in enumerate((x, gamma, beta)):
        g = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            args_p = [x.copy(), gamma.copy(), beta.copy()]
            args_m = [x.copy(), gamma.copy(), beta.copy()]
            args_p[arr_idx][idx] += h
            args_m[arr_idx][idx] -= h
            g[idx] = (loss(*args_p) - loss(*args_m)) / (2 * h)
        grads.append(g)
    return grads


def test_layernorm_backward_finite_difference(m):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 5))
    gamma = rng.normal(size=5)
    beta = rng.normal(size=5)
    G = rng.normal(size=(3, 5))  # loss = Σ y ⊙ G，所以 dy = G
    _, cache = m.layernorm_forward(x, gamma, beta, eps=1e-5)
    dx, dgamma, dbeta = m.layernorm_backward(G, cache)
    ndx, ndgamma, ndbeta = _numeric_grads(m, x, gamma, beta, G, eps=1e-5)
    np.testing.assert_allclose(dx, ndx, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(dgamma, ndgamma, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(dbeta, ndbeta, rtol=1e-5, atol=1e-7)


def test_layernorm_backward_shift_invariance(m):
    # LN(x + c·1) = LN(x)，所以 dx 沿 D 维求和恒为 0
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 7))
    _, cache = m.layernorm_forward(x, rng.normal(size=7), rng.normal(size=7))
    dx, _, _ = m.layernorm_backward(rng.normal(size=(4, 7)), cache)
    np.testing.assert_allclose(dx.sum(axis=1), 0.0, atol=1e-10)
