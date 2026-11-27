import math

import torch

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _close(actual, expected, atol=1e-12):
    torch.testing.assert_close(actual, expected if torch.is_tensor(expected) else _t(expected), atol=atol, rtol=0)


def _call(m, pc, pr, rc, rr, **kw):
    as_t = lambda x: x if torch.is_tensor(x) else _t(x)  # noqa: E731
    return m.dpo_loss(as_t(pc), as_t(pr), as_t(rc), as_t(rr), **kw)


def test_hand_computed(m):
    # h = 0.5 * ((-1 - -2) - (-3 - -2)) = 0.5 * (1 + 1) = 1 → loss = log(1 + e^{-1})
    loss, cr, rr, acc = _call(m, [-1.0], [-3.0], [-2.0], [-2.0], beta=0.5)
    _close(loss, math.log(1 + math.exp(-1)))
    _close(cr, [0.5])
    _close(rr, [-0.5])
    assert acc == 1.0


def test_policy_equals_ref_gives_log2(m):
    g = torch.Generator().manual_seed(0)
    pc, pr = torch.randn(4, generator=g, dtype=F64), torch.randn(4, generator=g, dtype=F64)
    loss, cr, rr, acc = _call(m, pc, pr, pc, pr, beta=0.1)
    _close(loss, math.log(2))
    _close(cr, torch.zeros(4, dtype=F64), atol=1e-15)
    assert acc == 0.0  # 严格大于才算对


def test_extreme_margins_are_stable(m):
    loss, *_ = _call(m, [1e4], [-1e4], [0.0], [0.0], beta=1.0)
    assert torch.isfinite(loss) and loss < 1e-12
    loss, *_ = _call(m, [-1e4], [1e4], [0.0], [0.0], beta=1.0)
    _close(loss, 2e4, atol=1e-8)


def test_batch_mean_and_accuracy(m):
    pc = [-1.0, -5.0, -2.0]
    pr = [-2.0, -1.0, -2.0]
    loss, cr, rr, acc = _call(m, pc, pr, [0.0] * 3, [0.0] * 3, beta=1.0)
    hs = [1.0, -4.0, 0.0]
    expected = sum(math.log(1 + math.exp(-h)) for h in hs) / 3
    _close(loss, expected)
    assert abs(acc - 1 / 3) < 1e-12


def test_label_smoothing(m):
    h = 0.8
    loss, *_ = _call(m, [h], [0.0], [0.0], [0.0], beta=1.0, label_smoothing=0.2)
    expected = 0.8 * math.log(1 + math.exp(-h)) + 0.2 * math.log(1 + math.exp(h))
    _close(loss, expected)
    a, *_ = _call(m, [h], [0.0], [0.0], [0.0], beta=1.0, label_smoothing=0.5)
    b, *_ = _call(m, [-h], [0.0], [0.0], [0.0], beta=1.0, label_smoothing=0.5)
    _close(a, b)  # ε=0.5 时对 h 对称


def test_gradient_signs_finite_difference(m):
    base = dict(pc=-4.0, pr=-3.0, rc=-3.5, rr=-3.2)

    def f(**over):
        a = dict(base, **over)
        return _call(m, [a["pc"]], [a["pr"]], [a["rc"]], [a["rr"]], beta=0.2)[0]

    e = 1e-5
    d_pc = (f(pc=base["pc"] + e) - f(pc=base["pc"] - e)) / (2 * e)
    d_pr = (f(pr=base["pr"] + e) - f(pr=base["pr"] - e)) / (2 * e)
    assert d_pc < 0, "提高 chosen 的 logprob 应降低损失"
    assert d_pr > 0, "提高 rejected 的 logprob 应提高损失"
    # 解析梯度：-β σ(-h)，h = 0.2 * ((-4+3.5) - (-3+3.2)) = -0.14
    h = 0.2 * ((-4.0 + 3.5) - (-3.0 + 3.2))
    _close(d_pc, -0.2 / (1 + math.exp(h)), atol=1e-8)
    _close(d_pr, 0.2 / (1 + math.exp(h)), atol=1e-8)


def test_autograd_matches_closed_form(m):
    # dL/dh = -(1-ε)σ(-h) + εσ(h)（每个样本），loss 是 batch 平均，所以
    # ∂L/∂pc = β·dL/dh / B，∂L/∂pr = -β·dL/dh / B；ε=0 时就是 -β·σ(-h)/B。参考模型不收梯度。
    g = torch.Generator().manual_seed(1)
    B, beta = 5, 0.3
    vals = [torch.randn(B, generator=g, dtype=F64) * 3 - 10 for _ in range(4)]
    h = beta * ((vals[0] - vals[2]) - (vals[1] - vals[3]))
    for ls in (0.0, 0.2):
        pc, pr, rc, rr = (v.clone().requires_grad_(True) for v in vals)
        loss, cr, rj, acc = m.dpo_loss(pc, pr, rc, rr, beta=beta, label_smoothing=ls)
        assert loss.dim() == 0 and loss.dtype == F64, "loss 应是 float64 的 0-d tensor"
        loss.backward()
        dh = -(1 - ls) * torch.sigmoid(-h) + ls * torch.sigmoid(h)
        _close(pc.grad, beta * dh / B)
        _close(pr.grad, -beta * dh / B)
        for name, t in (("ref_chosen_logps", rc), ("ref_rejected_logps", rr)):
            assert t.grad is None or torch.all(t.grad == 0), f"{name} 是常数，不该有梯度（detach）"
        assert not cr.requires_grad and not rj.requires_grad, "rewards 只用于日志，应 detach"
        assert isinstance(acc, float), "reward_accuracy 应是 python float"
