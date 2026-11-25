import math

import numpy as np


def _brute_logprob(row, y):
    mx = max(row)
    return float(row[y]) - (mx + math.log(math.fsum(math.exp(float(v) - mx) for v in row)))


def _brute_entropy(row):
    mx = max(row)
    z = math.fsum(math.exp(float(v) - mx) for v in row)
    ps = [math.exp(float(v) - mx) / z for v in row]
    return -math.fsum(p * math.log(p) for p in ps if p > 0)


def test_logprobs_brute_force(m):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 4, 7)) * 3
    labels = rng.integers(0, 7, size=(2, 4))
    lp = m.logprobs_from_logits(logits, labels)
    assert lp.shape == (2, 4)
    for b in range(2):
        for t in range(4):
            np.testing.assert_allclose(lp[b, t], _brute_logprob(logits[b, t], labels[b, t]), atol=1e-12)
    assert np.all(lp <= 0)


def test_logprobs_large_logits(m):
    logits = np.array([[[1000.0, 999.0, -1000.0]]])
    lp = m.logprobs_from_logits(logits, np.array([[1]]))
    np.testing.assert_allclose(lp[0, 0], -1.0 - math.log(1 + math.exp(-1.0)), atol=1e-12)
    lp2 = m.logprobs_from_logits(logits, np.array([[2]]))
    assert np.isfinite(lp2[0, 0]) and lp2[0, 0] < -1999


def test_entropy_brute_force_and_bounds(m):
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(3, 5, 6)) * 2
    ent = m.entropy_from_logits(logits)
    for b in range(3):
        for t in range(5):
            np.testing.assert_allclose(ent[b, t], _brute_entropy(logits[b, t]), atol=1e-12)
    assert np.all(ent >= -1e-12) and np.all(ent <= math.log(6) + 1e-12)


def test_entropy_extremes(m):
    uniform = np.zeros((1, 1, 10))
    np.testing.assert_allclose(m.entropy_from_logits(uniform)[0, 0], math.log(10), atol=1e-12)
    peaked = np.array([[[50.0, 0.0, 0.0, 0.0]]])
    assert m.entropy_from_logits(peaked)[0, 0] < 1e-18 + 4 * 51 * math.exp(-50)


def test_response_alignment(m):
    # 序列长 6，response 长 3（位置 3,4,5）。构造 logits：位置 t 的 logits 在 input_ids[t+1] 处是 5，其余 0
    V = 8
    input_ids = np.array([[1, 2, 3, 4, 5, 6], [7, 0, 2, 2, 1, 3]])
    logits = np.zeros((2, 6, V))
    for b in range(2):
        for t in range(5):
            logits[b, t, input_ids[b, t + 1]] = 5.0
    lp = m.response_logprobs(logits, input_ids, 3)
    assert lp.shape == (2, 3)
    expected = 5.0 - math.log(math.exp(5.0) + V - 1)
    np.testing.assert_allclose(lp, expected, atol=1e-12)
    # 错位一格的话会取到 0 的那一列，值完全不同
    assert not np.allclose(lp, -math.log(math.exp(5.0) + V - 1))


def test_response_matches_manual_slice(m):
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(2, 9, 11))
    ids = rng.integers(0, 11, size=(2, 9))
    lp = m.response_logprobs(logits, ids, 4)
    for b in range(2):
        for j, t in enumerate(range(5, 9)):  # response token 位置 5..8，用位置 4..7 的 logits
            np.testing.assert_allclose(lp[b, j], _brute_logprob(logits[b, t - 1], ids[b, t]), atol=1e-12)


def test_masked_mean(m):
    x = np.array([[1.0, 2.0, 100.0], [3.0, 1e9, 1e9]])
    mask = np.array([[1, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(m.masked_mean(x, mask), 2.0, atol=1e-12)
