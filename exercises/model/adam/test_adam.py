import math

import numpy as np


def test_first_step_is_sign_like(m):
    state = {}
    p = np.array([1.0, -2.0, 0.5])
    g = np.array([0.3, -40.0, 1e-3])
    new = m.adamw_step(p, g, state, lr=0.01)
    # 第 1 步 m̂ = g、v̂ = g²，更新 = lr * g / (|g| + eps)
    expected = p - 0.01 * g / (np.abs(g) + 1e-8)
    np.testing.assert_allclose(new, expected, atol=1e-12)
    assert state["step"] == 1
    np.testing.assert_allclose(state["m"], 0.1 * g, atol=1e-15)
    np.testing.assert_allclose(state["v"], 0.001 * g * g, atol=1e-15)


def test_second_step_hand_computed(m):
    state = {}
    p = np.array([0.0])
    p = m.adamw_step(p, np.array([1.0]), state, lr=0.1)
    p = m.adamw_step(p, np.array([-1.0]), state, lr=0.1)
    # m2 = 0.9·0.1 − 0.1 = −0.01；v2 = 0.999·0.001 + 0.001 = 0.001999
    # m̂ = −0.01/0.19；v̂ = 0.001999/0.001999 = 1
    step1 = -0.1 * 1.0 / (1.0 + 1e-8)
    step2 = -0.1 * (-0.01 / 0.19) / (math.sqrt(0.001999 / (1 - 0.999 ** 2)) + 1e-8)
    np.testing.assert_allclose(p, [step1 + step2], atol=1e-12)
    assert state["step"] == 2


def test_decoupled_weight_decay(m):
    state = {}
    p = np.array([2.0, -4.0])
    new = m.adamw_step(p, np.zeros(2), state, lr=0.1, weight_decay=0.5)
    np.testing.assert_allclose(new, p * (1 - 0.1 * 0.5), atol=1e-12)  # 梯度为 0 时只剩衰减
    state2 = {}
    g = np.array([1.0, 1.0])
    new2 = m.adamw_step(p, g, state2, lr=0.1, weight_decay=0.5)
    np.testing.assert_allclose(new2, p * 0.95 - 0.1 * g / (1 + 1e-8), atol=1e-12)


def test_gradient_scale_invariance(m):
    p = np.array([1.0, 2.0, 3.0])
    g = np.array([0.5, -0.2, 0.1])
    a = m.adamw_step(p, g, {}, lr=0.01)
    b = m.adamw_step(p, 1000 * g, {}, lr=0.01)
    np.testing.assert_allclose(a, b, atol=1e-8)  # 只差 eps 带来的 ~1e-9 量级


def test_state_is_updated_in_place(m):
    state = {}
    p = np.zeros((2, 3))
    for _ in range(3):
        p = m.adamw_step(p, np.ones((2, 3)), state)
    assert state["step"] == 3
    assert state["m"].shape == (2, 3) and state["v"].shape == (2, 3)


def test_converges_on_quadratic(m):
    a = np.array([1.0, 10.0, 0.1])
    x = np.array([5.0, -3.0, 2.0])
    state = {}
    for _ in range(3000):
        x = m.adamw_step(x, a * x, state, lr=0.02)  # ∇ ½Σ a x² = a ⊙ x
    assert np.all(np.abs(x) < 0.05), x
