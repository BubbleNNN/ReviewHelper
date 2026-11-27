import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _close(actual, expected, atol=1e-12):
    torch.testing.assert_close(actual, expected if torch.is_tensor(expected) else _t(expected), atol=atol, rtol=0)


def test_hand_values(m):
    lp, ref = _t([-1.0]), _t([-2.0])
    _close(m.kl_penalty(lp, ref, "k1"), [1.0])
    _close(m.kl_penalty(lp, ref, "kl"), [1.0])
    _close(m.kl_penalty(lp, ref, "abs"), [1.0])
    _close(m.kl_penalty(lp, ref, "k2"), [0.5])
    _close(m.kl_penalty(lp, ref, "mse"), [0.5])
    # u = ref - lp = -1 → e^{-1} + 1 - 1 = e^{-1}
    _close(m.kl_penalty(lp, ref, "k3"), [math.exp(-1)], atol=1e-15)
    _close(m.kl_penalty(lp, ref, "low_var_kl"), [math.exp(-1)], atol=1e-15)
    _close(m.kl_penalty(ref, lp, "k1"), [-1.0])  # k1 可以为负


def test_k3_clamps(m):
    out = m.kl_penalty(_t([0.0, 0.0, -3.0]), _t([-100.0, 100.0, -3.0]), "k3")
    _close(out, [10.0, 10.0, 0.0])
    assert torch.all(torch.isfinite(out))


def test_unknown_kind_raises(m):
    try:
        m.kl_penalty(torch.zeros(2, dtype=F64), torch.zeros(2, dtype=F64), "k4")
    except ValueError:
        return
    raise AssertionError("未知 kind 应抛 ValueError")


def _two_dists(seed, V=12, spread=0.3):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(V, generator=g, dtype=F64)
    p = torch.softmax(a, dim=0)
    b = a + spread * torch.randn(V, generator=g, dtype=F64)
    q = torch.softmax(b, dim=0)
    return p, q


def test_exact_kl(m):
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    _close(m.exact_kl(_t([0.5, 0.5]), _t([0.25, 0.75])), expected)
    _close(m.exact_kl(_t([0.2, 0.3, 0.5]), _t([0.2, 0.3, 0.5])), 0.0, atol=1e-15)
    _close(m.exact_kl(_t([0.0, 1.0]), _t([0.5, 0.5])), math.log(2))  # 0·log0 = 0
    p, q = _two_dists(5)
    _close(m.exact_kl(p, q), F.kl_div(q.log(), p, reduction="sum"))  # 内置函数当 oracle


def test_k1_and_k3_unbiased_monte_carlo(m):
    p, q = _two_dists(0)
    true_kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p.tolist(), q.tolist()))
    g = torch.Generator().manual_seed(1)
    x = torch.multinomial(p, 200_000, replacement=True, generator=g)  # x ~ π
    lp, ref = torch.log(p[x]), torch.log(q[x])
    for kind in ("k1", "k3"):
        est = m.kl_penalty(lp, ref, kind)
        mean, se = est.mean().item(), est.std().item() / math.sqrt(len(est))
        assert abs(mean - true_kl) < 5 * se + 1e-12, f"{kind} 均值 {mean:.5f} 与 KL {true_kl:.5f} 不符"


def test_k3_nonnegative_and_lower_variance(m):
    p, q = _two_dists(2, spread=0.2)
    g = torch.Generator().manual_seed(3)
    x = torch.multinomial(p, 50_000, replacement=True, generator=g)
    lp, ref = torch.log(p[x]), torch.log(q[x])
    k1 = m.kl_penalty(lp, ref, "k1")
    k3 = m.kl_penalty(lp, ref, "k3")
    k2 = m.kl_penalty(lp, ref, "k2")
    assert torch.all(k3 >= 0) and torch.all(k2 >= 0)
    assert torch.any(k1 < 0)
    assert k3.var() < 0.2 * k1.var(), "分布接近时 k3 的方差应远小于 k1"
    _close(k2, 0.5 * k1 ** 2)


def test_gradients_closed_form(m):
    # 逐元素导数：dk1/dℓ = 1，dk2/dℓ = ℓ - ℓ_ref，dk3/dℓ = 1 - e^{ℓ_ref - ℓ}（被 clamp 的位置为 0）
    lp0 = _t([-1.0, -2.0, -3.0, 0.0, 0.0])
    ref = _t([-2.0, -1.0, -3.0, -100.0, 100.0])  # 最后两个 k3 被 clamp 到 10
    expected = {
        "k1": torch.ones(5, dtype=F64),
        "k2": lp0 - ref,
        "k3": _t([1 - math.exp(-1), 1 - math.e, 0.0, 0.0, 0.0]),
    }
    for kind, g in expected.items():
        lp = lp0.clone().requires_grad_(True)
        out = m.kl_penalty(lp, ref, kind)
        assert out.dtype == F64 and out.shape == lp.shape
        out.sum().backward()
        assert lp.grad is not None, f"{kind}: 应能对 logprob 反传"
        _close(lp.grad, g)


def test_expected_gradient_k1_k2_k3(m):
    # π = softmax(θ)。把采到的 x 当成固定样本、只对 log π(x) 求导，再按 π(x) 加权，
    # 就是 E_{x~π}[∇θ k(x)]。结论：E[∇k1] = 0，E[∇k2] = ∇KL(π‖π_ref)，E[∇k3] = ∇KL(π_ref‖π)。
    p, q = _two_dists(4, V=8, spread=0.5)
    log_q = torch.log(q)

    def grad_theta(fn):
        theta = torch.log(p).clone().requires_grad_(True)
        fn(torch.log_softmax(theta, dim=0)).backward()
        return theta.grad

    def expected_of(kind):
        return lambda lp: (lp.exp().detach() * m.kl_penalty(lp, log_q, kind)).sum()

    _close(grad_theta(expected_of("k1")), torch.zeros(8, dtype=F64))
    forward_kl = grad_theta(lambda lp: (lp.exp() * (lp - log_q)).sum())
    _close(grad_theta(expected_of("k2")), forward_kl)
    reverse_kl = grad_theta(lambda lp: (q * (log_q - lp)).sum())
    _close(grad_theta(expected_of("k3")), reverse_kl)
