import numpy as np


def _ref_no_mask(r, v, gamma, lam):
    """按定义直接求和：A_t = Σ_k (γλ)^k δ_{t+k}"""
    T = len(r)
    v_next = np.append(v[1:], 0.0)
    delta = r + gamma * v_next - v
    return np.array([sum((gamma * lam) ** k * delta[t + k] for k in range(T - t)) for t in range(T)])


def test_hand_computed(m):
    # 手算：δ2=0.3, δ1=0.03, δ0=0.04；A2=0.3, A1=0.246, A0=0.21712
    r = np.array([[0.0, 0.0, 1.0]])
    v = np.array([[0.5, 0.6, 0.7]])
    adv, ret = m.compute_gae(r, v, np.ones((1, 3)), gamma=0.9, lam=0.8)
    np.testing.assert_allclose(adv, [[0.21712, 0.246, 0.3]], atol=1e-9)
    np.testing.assert_allclose(ret, [[0.71712, 0.846, 1.0]], atol=1e-9)


def test_matches_definition_random(m):
    rng = np.random.default_rng(0)
    r = rng.normal(size=(4, 9))
    v = rng.normal(size=(4, 9))
    adv, ret = m.compute_gae(r, v, np.ones_like(r), gamma=0.97, lam=0.9)
    for b in range(4):
        np.testing.assert_allclose(adv[b], _ref_no_mask(r[b], v[b], 0.97, 0.9), atol=1e-9)
    np.testing.assert_allclose(ret, adv + v, atol=1e-12)


def test_lambda_one_is_monte_carlo(m):
    rng = np.random.default_rng(1)
    r = rng.normal(size=(2, 6))
    v = rng.normal(size=(2, 6))
    g = 0.95
    adv, _ = m.compute_gae(r, v, np.ones_like(r), gamma=g, lam=1.0)
    mc = np.array([[sum(g ** k * r[b, t + k] for k in range(6 - t)) for t in range(6)] for b in range(2)])
    np.testing.assert_allclose(adv, mc - v, atol=1e-9)


def test_lambda_zero_is_td(m):
    rng = np.random.default_rng(2)
    r = rng.normal(size=(2, 5))
    v = rng.normal(size=(2, 5))
    adv, _ = m.compute_gae(r, v, np.ones_like(r), gamma=0.9, lam=0.0)
    v_next = np.concatenate([v[:, 1:], np.zeros((2, 1))], axis=1)
    np.testing.assert_allclose(adv, r + 0.9 * v_next - v, atol=1e-12)


def test_mask_skips_observation_tokens(m):
    # 有效 token 在位置 0,1,4,5；位置 2,3 是工具返回；6,7 是 padding
    rng = np.random.default_rng(3)
    r = rng.normal(size=(1, 8))
    v = rng.normal(size=(1, 8))
    mask = np.array([[1, 1, 0, 0, 1, 1, 0, 0]], dtype=float)
    adv, _ = m.compute_gae(r, v, mask, gamma=0.9, lam=0.7)
    keep = np.array([0, 1, 4, 5])
    expected = _ref_no_mask(r[0, keep], v[0, keep], 0.9, 0.7)
    np.testing.assert_allclose(adv[0, keep], expected, atol=1e-9)


def test_sparse_outcome_reward_gamma_one(m):
    # γ=λ=1 且只有末尾奖励时：A_t = R - V_t
    v = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.0, 0.0]])
    r = np.array([[0, 0, 0, 1.0], [0, 2.0, 0, 0]])
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=float)
    adv, _ = m.compute_gae(r, v, mask)
    np.testing.assert_allclose(adv[0], 1.0 - v[0], atol=1e-12)
    np.testing.assert_allclose(adv[1, :2], 2.0 - v[1, :2], atol=1e-12)
