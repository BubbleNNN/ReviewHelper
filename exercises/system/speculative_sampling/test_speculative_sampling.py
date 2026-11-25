import numpy as np

P = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6], [0.3, 0.3, 0.4]])
Q = np.array([[0.6, 0.2, 0.2], [0.1, 0.8, 0.1]])


def test_reject_second_resample(m):
    # 位置0：p/q = 0.5/0.6 ≈ 0.83 > 0.1 接受；位置1：p/q = 0.2/0.8 = 0.25 < 0.9 拒绝
    # 残差 max(0, p1-q1) = [0.1, 0, 0.5] → 归一化 [1/6, 0, 5/6]，u=0.5 → token 2
    assert m.verify(P, Q, [0, 1], [0.1, 0.9], 0.5) == [0, 2]


def test_all_accepted_bonus(m):
    # 都接受，bonus 从 p[2]=[0.3,0.3,0.4] 采：u=0.65 → 累积 [0.3,0.6,1.0] → token 2
    assert m.verify(P, Q, [0, 1], [0.1, 0.1], 0.65) == [0, 1, 2]
    assert m.verify(P, Q, [0, 1], [0.1, 0.1], 0.0) == [0, 1, 0]


def test_reject_first(m):
    # u=0.95 > 0.83 拒绝位置0；残差 [0, 0.1, 0] → 只能是 token 1
    out = m.verify(P, Q, [0, 1], [0.95, 0.0], 0.3)
    assert out == [1]


def test_ratio_above_one_always_accepted(m):
    # token 1 在位置0：p=0.3 > q=0.2，任何 u<1 都接受
    for u in (0.0, 0.5, 0.999999):
        out = m.verify(P, Q[:1], [1], [u], 0.2)
        assert out[0] == 1 and len(out) == 2


def test_first_token_distribution_is_exact(m):
    """用 u 的均匀网格做数值积分（确定性，非随机），检查第一个输出 token 的分布 = p[0]。"""
    rng = np.random.default_rng(3)
    N = 400
    grid = (np.arange(N) + 0.5) / N
    for trial in range(3):
        V = 5
        p = rng.dirichlet(np.ones(V), size=2)
        q = rng.dirichlet(np.ones(V), size=1)
        marg = np.zeros(V)
        for x in range(V):
            acc = np.mean([len(o) == 2 and o[0] == x for o in (m.verify(p, q, [x], [u], 0.5) for u in grid)])
            res = np.zeros(V)
            if acc < 1:
                u_rej = 1 - 0.25 / N  # 一定拒绝（此时 p/q < 1）
                for us in grid:
                    res[m.verify(p, q, [x], [u_rej], us)[0]] += 1.0 / N
            marg[x] += q[0, x] * acc
            marg += q[0, x] * (1 - acc) * res
        np.testing.assert_allclose(marg, p[0], atol=4.0 / N)


def test_bonus_distribution(m):
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(4), size=2)
    q = p[:1].copy()  # q == p 时一定接受
    N = 500
    grid = (np.arange(N) + 0.5) / N
    counts = np.zeros(4)
    for us in grid:
        out = m.verify(p, q, [2], [0.999], us)
        assert out[0] == 2 and len(out) == 2
        counts[out[1]] += 1.0 / N
    np.testing.assert_allclose(counts, p[1], atol=2.0 / N)


def test_expected_accept_length_monte_carlo(m):
    rng = np.random.default_rng(11)
    gamma, V = 3, 6
    p = rng.dirichlet(np.ones(V) * 0.7, size=gamma + 1)
    q = rng.dirichlet(np.ones(V) * 0.7, size=gamma)
    alpha = np.minimum(p[:gamma], q).sum(axis=1)  # 每个位置的接受率
    expected = sum(np.prod(alpha[:k]) for k in range(1, gamma + 1))
    T = 4000
    drafts = np.stack([rng.choice(V, size=T, p=q[i]) for i in range(gamma)], axis=1)
    ua = rng.random((T, gamma))
    us = rng.random(T)
    total = 0
    for t in range(T):
        out = m.verify(p, q, [int(x) for x in drafts[t]], ua[t], us[t])
        assert 1 <= len(out) <= gamma + 1
        total += len(out) - 1
    assert abs(total / T - expected) < 0.06
