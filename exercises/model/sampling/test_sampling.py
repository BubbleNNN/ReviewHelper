import math

import numpy as np


def test_top_k_hand(m):
    p = np.array([0.1, 0.4, 0.3, 0.2])
    np.testing.assert_allclose(m.top_k_filter(p, 2), [0, 4 / 7, 3 / 7, 0], atol=1e-12)
    np.testing.assert_allclose(m.top_k_filter(p, 1), [0, 1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(m.top_k_filter(p, 0), p, atol=1e-12)
    np.testing.assert_allclose(m.top_k_filter(p, 10), p, atol=1e-12)


def test_top_k_ties_kept(m):
    p = np.array([0.4, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(m.top_k_filter(p, 2), p, atol=1e-12)  # 第 2 名有三个并列，全保留


def test_top_p_minimal_prefix(m):
    p = np.array([0.15, 0.5, 0.05, 0.3])  # 降序：0.5, 0.3, 0.15, 0.05；累计 0.5, 0.8, 0.95, 1.0
    np.testing.assert_allclose(m.top_p_filter(p, 0.75), [0, 0.625, 0, 0.375], atol=1e-12)
    np.testing.assert_allclose(m.top_p_filter(p, 0.81), [0.15 / 0.95, 0.5 / 0.95, 0, 0.3 / 0.95], atol=1e-12)
    np.testing.assert_allclose(m.top_p_filter(p, 0.3), [0, 1, 0, 0], atol=1e-12)  # 至少保留 1 个
    np.testing.assert_allclose(m.top_p_filter(p, 1.0), p, atol=1e-12)


def test_top_p_stable_tie_break(m):
    p = np.array([0.2, 0.4, 0.2, 0.2])
    np.testing.assert_allclose(m.top_p_filter(p, 0.5), [1 / 3, 2 / 3, 0, 0], atol=1e-12)


def test_temperature_and_order(m):
    logits = np.array([1.0, 2.0, 3.0])
    z = [math.exp(x / 2.0) for x in logits]
    np.testing.assert_allclose(m.process_logits(logits, temperature=2.0), np.array(z) / sum(z), atol=1e-12)
    # greedy
    np.testing.assert_allclose(m.process_logits(np.array([0.5, 3.0, 3.0, 1.0]), temperature=0.0), [0, 1, 0, 0])
    sharp = m.process_logits(logits, temperature=1e-3)
    assert sharp[2] > 1 - 1e-12
    # 先 top-k 再 top-p：probs∝e^[0,1,2,3]，top_k=3 → [e,e²,e³]/Σ，再 top_p=0.7
    lg = np.array([0.0, 1.0, 2.0, 3.0])
    e = np.exp([1.0, 2.0, 3.0])
    pk = e / e.sum()  # ≈ [0.090, 0.245, 0.665]
    assert pk[2] < 0.7 < pk[2] + pk[1]
    expect = np.array([0.0, 0.0, pk[1], pk[2]]) / (pk[1] + pk[2])
    np.testing.assert_allclose(m.process_logits(lg, top_k=3, top_p=0.7), expect, atol=1e-12)


def test_sampler_frequencies(m):
    rng = np.random.default_rng(0)
    logits = np.array([0.0, 1.0, 2.0, -1.0])
    n = 4000
    counts = np.zeros(4)
    for _ in range(n):
        counts[m.sample(logits, rng, top_k=3)] += 1
    e = np.exp([0.0, 1.0, 2.0])
    target = np.append(e / e.sum(), 0.0)
    assert counts[3] == 0, "top_k=3 时第 4 个 token 永远不该被采到"
    np.testing.assert_allclose(counts / n, target, atol=0.035)


def test_sampler_uses_given_rng_and_greedy(m):
    logits = np.array([0.3, -1.0, 2.5, 0.0])
    a = [m.sample(logits, np.random.default_rng(7)) for _ in range(5)]
    b = [m.sample(logits, np.random.default_rng(7)) for _ in range(5)]
    assert a == b, "同一个种子应得到同样的结果"
    rng = np.random.default_rng(1)
    assert all(m.sample(logits, rng, temperature=0.0) == 2 for _ in range(20))
    assert all(m.sample(logits, rng, top_k=1) == 2 for _ in range(20))
    assert isinstance(m.sample(logits, rng), int)
