import math

import numpy as np


def test_router_probs(m):
    p = m.router_probs(np.array([[0.0, 0.0], [1000.0, 0.0]]))
    np.testing.assert_allclose(p, [[0.5, 0.5], [1.0, 0.0]], atol=1e-12)


def test_top_k_hand(m):
    idx, w = m.top_k_routing(np.array([[1.0, 2.0, 3.0, 0.0], [0.5, 0.5, 0.0, 0.5]]), 2)
    np.testing.assert_array_equal(idx[0], [2, 1])
    s = 1 / (1 + math.exp(-1))  # e³/(e³+e²) = σ(1)
    np.testing.assert_allclose(w[0], [s, 1 - s], atol=1e-12)
    np.testing.assert_array_equal(idx[1], [0, 1])  # 三个并列，取下标小的
    np.testing.assert_allclose(w[1], [0.5, 0.5], atol=1e-12)
    idx1, w1 = m.top_k_routing(np.array([[0.1, 0.9, 0.3]]), 1)
    np.testing.assert_array_equal(idx1, [[1]])
    np.testing.assert_allclose(w1, [[1.0]])


def test_capacity(m):
    assert m.expert_capacity(10, 4, 2, 1.0) == 5
    assert m.expert_capacity(10, 4, 2, 1.25) == 7   # ceil(6.25)
    assert m.expert_capacity(16, 8, 1, 1.0) == 2


def test_dispatch_slot_priority(m):
    idx = np.array([[0, 1], [0, 2], [0, 1], [1, 0]])
    assign, dropped = m.dispatch(idx, 3, capacity=2)
    # 第 1 轮（slot 0）：t0→e0, t1→e0, t2→e0 满 → 丢, t3→e1
    # 第 2 轮（slot 1）：t0→e1, t1→e2, t2→e1 满 → 丢, t3→e0 满 → 丢
    assert [list(a) for a in assign] == [[0, 1], [3, 0], [1]]
    np.testing.assert_array_equal(dropped, [[False, False], [False, False], [True, True], [False, True]])


def test_aux_loss_uniform_and_collapsed(m):
    logits = np.zeros((8, 4))
    idx = np.array([[0], [1], [2], [3], [0], [1], [2], [3]])
    np.testing.assert_allclose(m.load_balancing_loss(logits, idx, 4), 1.0, atol=1e-12)
    logits2 = np.tile(np.log([0.9, 0.1]), (4, 1))
    idx2 = np.zeros((4, 1), dtype=int)
    np.testing.assert_allclose(m.load_balancing_loss(logits2, idx2, 2), 1.8, atol=1e-12)  # 2·(1·0.9 + 0·0.1)


def test_aux_loss_counts_all_k_slots(m):
    logits = np.zeros((2, 4))  # P = 1/4
    idx = np.array([[0, 1], [0, 2]])  # f = [2/4, 1/4, 1/4, 0]
    np.testing.assert_allclose(m.load_balancing_loss(logits, idx, 4), 4 * 0.25 * 1.0, atol=1e-12)


def test_moe_forward_combine_and_drop(m):
    rng = np.random.default_rng(0)
    T, d, E = 6, 3, 3
    x = rng.normal(size=(T, d))
    logits = rng.normal(size=(T, E))
    experts = [lambda z, c=c: z * (c + 1.0) for c in range(E)]  # 专家 e 把输入乘 e+1
    idx, w = m.top_k_routing(logits, 2)
    y = m.moe_forward(x, logits, experts, k=2, capacity=100)
    expected = np.stack([(w[t, 0] * (idx[t, 0] + 1) + w[t, 1] * (idx[t, 1] + 1)) * x[t] for t in range(T)])
    np.testing.assert_allclose(y, expected, atol=1e-12)
    # 容量 1：每个专家只收 1 个 token，其余 slot 贡献 0
    y1 = m.moe_forward(x, logits, experts, k=2, capacity=1)
    _, dropped = m.dispatch(idx, E, 1)
    exp1 = np.stack([sum(0.0 if dropped[t, j] else w[t, j] * (idx[t, j] + 1) for j in range(2)) * x[t]
                     for t in range(T)])
    np.testing.assert_allclose(y1, exp1, atol=1e-12)
    assert dropped.sum() == 2 * T - len(set(idx.ravel().tolist()))  # 每个被选中的专家恰好收 1 个
