import math

import numpy as np


def test_hand_values(m):
    lp, ref = np.array([-1.0]), np.array([-2.0])
    np.testing.assert_allclose(m.kl_penalty(lp, ref, "k1"), [1.0])
    np.testing.assert_allclose(m.kl_penalty(lp, ref, "kl"), [1.0])
    np.testing.assert_allclose(m.kl_penalty(lp, ref, "abs"), [1.0])
    np.testing.assert_allclose(m.kl_penalty(lp, ref, "k2"), [0.5])
    np.testing.assert_allclose(m.kl_penalty(lp, ref, "mse"), [0.5])
    # u = ref - lp = -1 → e^{-1} + 1 - 1 = e^{-1}
    np.testing.assert_allclose(m.kl_penalty(lp, ref, "k3"), [math.exp(-1)], atol=1e-15)
    np.testing.assert_allclose(m.kl_penalty(lp, ref, "low_var_kl"), [math.exp(-1)], atol=1e-15)
    np.testing.assert_allclose(m.kl_penalty(ref, lp, "k1"), [-1.0])  # k1 可以为负


def test_k3_clamps(m):
    out = m.kl_penalty(np.array([0.0, 0.0, -3.0]), np.array([-100.0, 100.0, -3.0]), "k3")
    np.testing.assert_allclose(out, [10.0, 10.0, 0.0], atol=1e-12)
    assert np.all(np.isfinite(out))


def test_unknown_kind_raises(m):
    try:
        m.kl_penalty(np.zeros(2), np.zeros(2), "k4")
    except ValueError:
        return
    raise AssertionError("未知 kind 应抛 ValueError")


def test_exact_kl(m):
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    np.testing.assert_allclose(m.exact_kl([0.5, 0.5], [0.25, 0.75]), expected, atol=1e-12)
    np.testing.assert_allclose(m.exact_kl([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0.0, atol=1e-15)
    np.testing.assert_allclose(m.exact_kl([0.0, 1.0], [0.5, 0.5]), math.log(2), atol=1e-12)  # 0·log0 = 0


def _two_dists(seed, V=12, spread=0.3):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=V)
    p = np.exp(a) / np.exp(a).sum()
    b = a + spread * rng.normal(size=V)
    q = np.exp(b) / np.exp(b).sum()
    return p, q


def test_k1_and_k3_unbiased_monte_carlo(m):
    p, q = _two_dists(0)
    true_kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    rng = np.random.default_rng(1)
    x = rng.choice(len(p), size=200_000, p=p)  # x ~ π
    lp, ref = np.log(p[x]), np.log(q[x])
    for kind in ("k1", "k3"):
        est = m.kl_penalty(lp, ref, kind)
        se = est.std() / math.sqrt(len(est))
        assert abs(est.mean() - true_kl) < 5 * se + 1e-12, f"{kind} 均值 {est.mean():.5f} 与 KL {true_kl:.5f} 不符"


def test_k3_nonnegative_and_lower_variance(m):
    p, q = _two_dists(2, spread=0.2)
    rng = np.random.default_rng(3)
    x = rng.choice(len(p), size=50_000, p=p)
    lp, ref = np.log(p[x]), np.log(q[x])
    k1 = m.kl_penalty(lp, ref, "k1")
    k3 = m.kl_penalty(lp, ref, "k3")
    k2 = m.kl_penalty(lp, ref, "k2")
    assert np.all(k3 >= 0) and np.all(k2 >= 0)
    assert np.any(k1 < 0)
    assert k3.var() < 0.2 * k1.var(), "分布接近时 k3 的方差应远小于 k1"
    np.testing.assert_allclose(k2, 0.5 * k1 ** 2, atol=1e-12)
