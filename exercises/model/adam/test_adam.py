import math

import torch

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _torch_adamw(p0, grads, lr, betas, eps, weight_decay):
    """参照答案：torch.optim.AdamW 在同一串梯度上逐步更新，返回每一步后的 (param, exp_avg, exp_avg_sq)。"""
    p = torch.nn.Parameter(p0.clone())
    opt = torch.optim.AdamW([p], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    out = []
    for g in grads:
        p.grad = g.clone()
        opt.step()
        st = opt.state[p]
        out.append((p.detach().clone(), st["exp_avg"].clone(), st["exp_avg_sq"].clone()))
    return out


def test_first_step_is_sign_like(m):
    state = {}
    p = _t([1.0, -2.0, 0.5])
    g = _t([0.3, -40.0, 1e-3])
    new = m.adamw_step(p, g, state, lr=0.01)
    # 第 1 步 m̂ = g、v̂ = g²，更新 = lr * g / (|g| + eps)
    expected = p - 0.01 * g / (g.abs() + 1e-8)
    _close(new, expected, atol=1e-12)
    assert state["step"] == 1
    _close(state["m"], 0.1 * g, atol=1e-12)
    _close(state["v"], 0.001 * g * g, atol=1e-12)  # (1-0.999) 与 0.001 在 float64 下差 ~1e-15 量级


def test_second_step_hand_computed(m):
    state = {}
    p = _t([0.0])
    p = m.adamw_step(p, _t([1.0]), state, lr=0.1)
    p = m.adamw_step(p, _t([-1.0]), state, lr=0.1)
    # m2 = 0.9·0.1 − 0.1 = −0.01；v2 = 0.999·0.001 + 0.001 = 0.001999
    # m̂ = −0.01/0.19；v̂ = 0.001999/0.001999 = 1
    step1 = -0.1 * 1.0 / (1.0 + 1e-8)
    step2 = -0.1 * (-0.01 / 0.19) / (math.sqrt(0.001999 / (1 - 0.999 ** 2)) + 1e-8)
    _close(p, _t([step1 + step2]), atol=1e-12)
    assert state["step"] == 2


def test_decoupled_weight_decay(m):
    state = {}
    p = _t([2.0, -4.0])
    new = m.adamw_step(p, torch.zeros(2, dtype=F64), state, lr=0.1, weight_decay=0.5)
    _close(new, p * (1 - 0.1 * 0.5), atol=1e-12)  # 梯度为 0 时只剩衰减
    state2 = {}
    g = _t([1.0, 1.0])
    new2 = m.adamw_step(p, g, state2, lr=0.1, weight_decay=0.5)
    _close(new2, p * 0.95 - 0.1 * g / (1 + 1e-8), atol=1e-12)


def test_gradient_scale_invariance(m):
    p = _t([1.0, 2.0, 3.0])
    g = _t([0.5, -0.2, 0.1])
    a = m.adamw_step(p, g, {}, lr=0.01)
    b = m.adamw_step(p, 1000 * g, {}, lr=0.01)
    _close(a, b, atol=1e-8)  # 只差 eps 带来的 ~1e-9 量级


def test_state_is_updated_in_place(m):
    state = {}
    p = torch.zeros(2, 3, dtype=F64)
    for _ in range(3):
        p = m.adamw_step(p, torch.ones(2, 3, dtype=F64), state)
    assert state["step"] == 3
    assert state["m"].shape == (2, 3) and state["v"].shape == (2, 3)


def test_converges_on_quadratic(m):
    a = _t([1.0, 10.0, 0.1])
    x = _t([5.0, -3.0, 2.0])
    state = {}
    for _ in range(3000):
        x = m.adamw_step(x, a * x, state, lr=0.02)  # ∇ ½Σ a x² = a ⊙ x
    assert bool((x.abs() < 0.05).all()), x


def test_matches_torch_adamw(m):
    # 和 torch.optim.AdamW 逐步对拍：非默认 betas、权重衰减、较大的 eps（检查 eps 加在根号外）
    configs = [
        dict(lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, steps=8),
        dict(lr=0.05, betas=(0.8, 0.99), eps=1e-2, weight_decay=0.0, steps=5),
        dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, steps=3),  # 本题默认值（torch 默认 wd=0.01，要显式传 0）
    ]
    for ci, cfg in enumerate(configs):
        steps = cfg.pop("steps")
        g = torch.Generator().manual_seed(ci)
        p0 = torch.randn(3, 4, generator=g, dtype=F64)
        grads = [torch.randn(3, 4, generator=g, dtype=F64) * (k + 1) for k in range(steps)]
        state, p, mine = {}, p0.clone(), []
        for gr in grads:
            p = m.adamw_step(p, gr, state, **cfg)
            mine.append((p, state["m"].clone(), state["v"].clone(), state["step"]))
        ref = _torch_adamw(p0, grads, **cfg)
        for k, ((pm, mm, vm, step), (pr, mr, vr)) in enumerate(zip(mine, ref)):
            assert step == k + 1, f"config {ci}: 第 {k + 1} 次调用后 state['step'] 应为 {k + 1}"
            _close(pm, pr, atol=1e-12)
            _close(mm, mr, atol=1e-12)
            _close(vm, vr, atol=1e-12)


def test_float32_dtype_preserved(m):
    # 输出和 state 的 dtype 要跟着输入走，不能偷偷转成 float64
    g = torch.Generator().manual_seed(7)
    p0 = torch.randn(5, generator=g)  # float32
    grads = [torch.randn(5, generator=g) for _ in range(3)]
    state, p = {}, p0.clone()
    for gr in grads:
        p = m.adamw_step(p, gr, state, lr=0.01, weight_decay=0.05)
    assert p.dtype == torch.float32, "返回的参数 dtype 应与输入一致"
    assert state["m"].dtype == torch.float32 and state["v"].dtype == torch.float32, "m / v 的 dtype 应与 param 一致"
    ref_p, ref_m, ref_v = _torch_adamw(p0, grads, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)[-1]
    _close(p, ref_p, atol=1e-6)
    _close(state["m"], ref_m, atol=1e-6)
    _close(state["v"], ref_v, atol=1e-6)
