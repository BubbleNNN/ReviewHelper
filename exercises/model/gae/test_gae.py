import torch

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _randn(g, *shape):
    return torch.randn(*shape, generator=g, dtype=F64)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _ref_no_mask(r, v, gamma, lam):
    """按定义直接求和：A_t = Σ_k (γλ)^k δ_{t+k}（r, v 是一维 tensor）"""
    T = len(r)
    v_next = torch.cat([v[1:], v.new_zeros(1)])
    delta = r + gamma * v_next - v
    return torch.stack([sum((gamma * lam) ** k * delta[t + k] for k in range(T - t)) for t in range(T)])


def _ones(*shape):
    return torch.ones(*shape, dtype=torch.bool)


def test_hand_computed(m):
    # 手算：δ2=0.3, δ1=0.03, δ0=0.04；A2=0.3, A1=0.246, A0=0.21712
    r = _t([[0.0, 0.0, 1.0]])
    v = _t([[0.5, 0.6, 0.7]])
    adv, ret = m.compute_gae(r, v, _ones(1, 3), gamma=0.9, lam=0.8)
    _close(adv, _t([[0.21712, 0.246, 0.3]]), atol=1e-9)
    _close(ret, _t([[0.71712, 0.846, 1.0]]), atol=1e-9)


def test_matches_definition_random(m):
    g = torch.Generator().manual_seed(0)
    r = _randn(g, 4, 9)
    v = _randn(g, 4, 9)
    adv, ret = m.compute_gae(r, v, _ones(4, 9), gamma=0.97, lam=0.9)
    for b in range(4):
        _close(adv[b], _ref_no_mask(r[b], v[b], 0.97, 0.9), atol=1e-9)
    _close(ret, adv + v, atol=1e-12)


def test_lambda_one_is_monte_carlo(m):
    g = torch.Generator().manual_seed(1)
    r = _randn(g, 2, 6)
    v = _randn(g, 2, 6)
    gm = 0.95
    adv, _ = m.compute_gae(r, v, _ones(2, 6), gamma=gm, lam=1.0)
    mc = torch.stack([torch.stack([sum(gm ** k * r[b, t + k] for k in range(6 - t)) for t in range(6)]) for b in range(2)])
    _close(adv, mc - v, atol=1e-9)


def test_lambda_zero_is_td(m):
    g = torch.Generator().manual_seed(2)
    r = _randn(g, 2, 5)
    v = _randn(g, 2, 5)
    adv, _ = m.compute_gae(r, v, _ones(2, 5), gamma=0.9, lam=0.0)
    v_next = torch.cat([v[:, 1:], v.new_zeros(2, 1)], dim=1)
    _close(adv, r + 0.9 * v_next - v, atol=1e-12)


def test_mask_skips_observation_tokens(m):
    # 有效 token 在位置 0,1,4,5；位置 2,3 是工具返回；6,7 是 padding
    g = torch.Generator().manual_seed(3)
    r = _randn(g, 1, 8)
    v = _randn(g, 1, 8)
    mask = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.bool)
    adv, _ = m.compute_gae(r, v, mask, gamma=0.9, lam=0.7)
    keep = torch.tensor([0, 1, 4, 5])
    expected = _ref_no_mask(r[0, keep], v[0, keep], 0.9, 0.7)
    _close(adv[0, keep], expected, atol=1e-9)


def test_sparse_outcome_reward_gamma_one(m):
    # γ=λ=1 且只有末尾奖励时：A_t = R - V_t
    v = _t([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.0, 0.0]])
    r = _t([[0, 0, 0, 1.0], [0, 2.0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    adv, _ = m.compute_gae(r, v, mask)
    _close(adv[0], 1.0 - v[0], atol=1e-12)
    _close(adv[1, :2], 2.0 - v[1, :2], atol=1e-12)


def test_no_grad_and_dtype(m):
    # values 来自 critic、带梯度；returns 要当 value loss 的目标，必须不带梯度
    g = torch.Generator().manual_seed(4)
    r = torch.randn(2, 5, generator=g)  # float32
    v = torch.randn(2, 5, generator=g).requires_grad_(True)
    adv, ret = m.compute_gae(r, v, _ones(2, 5), gamma=0.99, lam=0.95)
    assert adv.dtype == torch.float32 and ret.dtype == torch.float32, "输出 dtype 应与输入一致"
    assert not adv.requires_grad and not ret.requires_grad, "advantages / returns 不应带梯度（用 torch.no_grad 或 detach）"
