import math

import numpy as np


def _softmax_ref(row):
    """按定义直接算，只用于数值不大的输入。"""
    e = [math.exp(float(v)) for v in row]
    s = math.fsum(e)
    return [x / s for x in e]


def test_small_values_match_definition(m):
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(m.softmax(x), _softmax_ref([1, 2, 3]), rtol=1e-12)
    expected_lse = math.log(math.exp(1) + math.exp(2) + math.exp(3))
    np.testing.assert_allclose(m.logsumexp(x), expected_lse, rtol=1e-12)


def test_huge_logits_do_not_overflow(m):
    x = np.array([1e4, 1e4 + 1.0, 1e4 - 1.0])
    p = m.softmax(x)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p, _softmax_ref([0.0, 1.0, -1.0]), rtol=1e-9)
    expected = 1e4 + math.log(math.exp(0.0) + math.exp(1.0) + math.exp(-1.0))
    np.testing.assert_allclose(m.logsumexp(x), expected, rtol=1e-12)


def test_very_negative_logits_do_not_underflow(m):
    x = np.array([-1000.0, -1000.0])
    np.testing.assert_allclose(m.logsumexp(x), -1000.0 + math.log(2.0), rtol=1e-12)
    np.testing.assert_allclose(m.softmax(x), [0.5, 0.5], rtol=1e-12)
    np.testing.assert_allclose(m.log_softmax(x), [-math.log(2.0)] * 2, rtol=1e-12)


def test_shift_invariance(m):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 7))
    np.testing.assert_allclose(m.softmax(x), m.softmax(x + 123.0), atol=1e-12)
    np.testing.assert_allclose(m.logsumexp(x + 5.0), m.logsumexp(x) + 5.0, atol=1e-12)


def test_neg_inf_mask(m):
    x = np.array([0.0, -np.inf, 0.0, -np.inf])
    np.testing.assert_allclose(m.softmax(x), [0.5, 0.0, 0.5, 0.0], atol=1e-15)
    ls = m.log_softmax(x)
    assert ls[1] == -np.inf and ls[3] == -np.inf
    np.testing.assert_allclose(ls[[0, 2]], [-math.log(2.0)] * 2, rtol=1e-12)
    out = m.logsumexp(np.array([[-np.inf, -np.inf], [0.0, -np.inf]]))
    assert out[0] == -np.inf, "整行 -inf 时 logsumexp 应为 -inf，而不是 nan"
    np.testing.assert_allclose(out[1], 0.0, atol=1e-15)


def test_axis_and_keepdims(m):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 5))
    out0 = m.logsumexp(x, axis=0)
    assert out0.shape == (5,)
    for j in range(5):
        ref = math.log(math.fsum(math.exp(v) for v in x[:, j]))
        np.testing.assert_allclose(out0[j], ref, rtol=1e-12)
    assert m.logsumexp(x, axis=1, keepdims=True).shape == (3, 1)
    p = m.softmax(x, axis=0)
    assert p.shape == x.shape
    np.testing.assert_allclose(p.sum(axis=0), np.ones(5), rtol=1e-12)


def test_log_softmax_consistent_with_softmax(m):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 6)) * 3
    ls = m.log_softmax(x)
    for b in range(2):
        np.testing.assert_allclose(np.exp(ls[b]), _softmax_ref(x[b]), rtol=1e-10)
    # 概率极小（e^-5000）的位置：log(softmax) 会变成 -inf，正确实现应给出有限值
    big = np.array([[5e3, 0.0, -5e3]])
    lb = m.log_softmax(big)
    assert np.all(np.isfinite(lb))
    np.testing.assert_allclose(lb[0], [0.0, -5e3, -1e4], rtol=1e-12, atol=1e-12)
