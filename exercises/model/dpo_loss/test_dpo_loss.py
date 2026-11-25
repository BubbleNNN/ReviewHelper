import math

import numpy as np


def _call(m, pc, pr, rc, rr, **kw):
    return m.dpo_loss(np.array(pc, float), np.array(pr, float), np.array(rc, float), np.array(rr, float), **kw)


def test_hand_computed(m):
    # h = 0.5 * ((-1 - -2) - (-3 - -2)) = 0.5 * (1 + 1) = 1 → loss = log(1 + e^{-1})
    loss, cr, rr, acc = _call(m, [-1.0], [-3.0], [-2.0], [-2.0], beta=0.5)
    np.testing.assert_allclose(loss, math.log(1 + math.exp(-1)), atol=1e-12)
    np.testing.assert_allclose(cr, [0.5], atol=1e-12)
    np.testing.assert_allclose(rr, [-0.5], atol=1e-12)
    assert acc == 1.0


def test_policy_equals_ref_gives_log2(m):
    rng = np.random.default_rng(0)
    pc, pr = rng.normal(size=4), rng.normal(size=4)
    loss, cr, rr, acc = _call(m, pc, pr, pc, pr, beta=0.1)
    np.testing.assert_allclose(loss, math.log(2), atol=1e-12)
    np.testing.assert_allclose(cr, 0.0, atol=1e-15)
    assert acc == 0.0  # 严格大于才算对


def test_extreme_margins_are_stable(m):
    loss, *_ = _call(m, [1e4], [-1e4], [0.0], [0.0], beta=1.0)
    assert math.isfinite(loss) and loss < 1e-12
    loss, *_ = _call(m, [-1e4], [1e4], [0.0], [0.0], beta=1.0)
    np.testing.assert_allclose(loss, 2e4, rtol=1e-12)


def test_batch_mean_and_accuracy(m):
    pc = [-1.0, -5.0, -2.0]
    pr = [-2.0, -1.0, -2.0]
    loss, cr, rr, acc = _call(m, pc, pr, [0.0] * 3, [0.0] * 3, beta=1.0)
    hs = [1.0, -4.0, 0.0]
    expected = sum(math.log(1 + math.exp(-h)) for h in hs) / 3
    np.testing.assert_allclose(loss, expected, atol=1e-12)
    np.testing.assert_allclose(acc, 1 / 3, atol=1e-12)


def test_label_smoothing(m):
    h = 0.8
    loss, *_ = _call(m, [h], [0.0], [0.0], [0.0], beta=1.0, label_smoothing=0.2)
    expected = 0.8 * math.log(1 + math.exp(-h)) + 0.2 * math.log(1 + math.exp(h))
    np.testing.assert_allclose(loss, expected, atol=1e-12)
    a, *_ = _call(m, [h], [0.0], [0.0], [0.0], beta=1.0, label_smoothing=0.5)
    b, *_ = _call(m, [-h], [0.0], [0.0], [0.0], beta=1.0, label_smoothing=0.5)
    np.testing.assert_allclose(a, b, atol=1e-12)  # ε=0.5 时对 h 对称


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
    np.testing.assert_allclose(d_pc, -0.2 / (1 + math.exp(h)), rtol=1e-6)
    np.testing.assert_allclose(d_pr, 0.2 / (1 + math.exp(h)), rtol=1e-6)
