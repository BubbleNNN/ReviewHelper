import torch

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _L(x):
    return torch.tensor(x, dtype=torch.long)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _rand_dist(g, rows, V):
    """随机概率分布，每行服从 Dirichlet(1)（归一化的 Exp(1)），和为 1。"""
    e = -torch.log(torch.rand(rows, V, generator=g, dtype=F64))
    return e / e.sum(dim=-1, keepdim=True)


P = _t([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6], [0.3, 0.3, 0.4]])
Q = _t([[0.6, 0.2, 0.2], [0.1, 0.8, 0.1]])


def test_reject_second_resample(m):
    # 位置0：p/q = 0.5/0.6 ≈ 0.83 > 0.1 接受；位置1：p/q = 0.2/0.8 = 0.25 < 0.9 拒绝
    # 残差 max(0, p1-q1) = [0.1, 0, 0.5] → 归一化 [1/6, 0, 5/6]，u=0.5 → token 2
    out = m.verify(P, Q, _L([0, 1]), _t([0.1, 0.9]), 0.5)
    assert out == [0, 2]
    assert all(isinstance(t, int) for t in out), "verify 应返回 python int 的 list（用 int(...) 转换）"


def test_all_accepted_bonus(m):
    # 都接受，bonus 从 p[2]=[0.3,0.3,0.4] 采：u=0.65 → 累积 [0.3,0.6,1.0] → token 2
    assert m.verify(P, Q, _L([0, 1]), _t([0.1, 0.1]), 0.65) == [0, 1, 2]
    assert m.verify(P, Q, _L([0, 1]), _t([0.1, 0.1]), 0.0) == [0, 1, 0]


def test_reject_first(m):
    # u=0.95 > 0.83 拒绝位置0；残差 [0, 0.1, 0] → 只能是 token 1
    out = m.verify(P, Q, _L([0, 1]), _t([0.95, 0.0]), 0.3)
    assert out == [1]


def test_ratio_above_one_always_accepted(m):
    # token 1 在位置0：p=0.3 > q=0.2，任何 u<1 都接受
    for u in (0.0, 0.5, 0.999999):
        out = m.verify(P, Q[:1], _L([1]), _t([u]), 0.2)
        assert out[0] == 1 and len(out) == 2


def test_first_token_distribution_is_exact(m):
    """用 u 的均匀网格做数值积分（确定性，非随机），检查第一个输出 token 的分布 = p[0]。"""
    g = torch.Generator().manual_seed(3)
    N = 400
    grid = (torch.arange(N, dtype=F64) + 0.5) / N
    u_rows = [grid[i : i + 1] for i in range(N)]  # 预先建好 (1,) 的 u_accept，减少循环开销
    u_list = grid.tolist()
    u_rej = _t([1 - 0.25 / N])  # 一定拒绝（此时 p/q < 1）
    for trial in range(3):
        V = 5
        p = _rand_dist(g, 2, V)
        q = _rand_dist(g, 1, V)
        toks = [_L([x]) for x in range(V)]
        marg = torch.zeros(V, dtype=F64)
        for x in range(V):
            acc = sum(len(o) == 2 and o[0] == x for o in (m.verify(p, q, toks[x], ua, 0.5) for ua in u_rows)) / N
            res = torch.zeros(V, dtype=F64)
            if acc < 1:
                for us in u_list:
                    res[m.verify(p, q, toks[x], u_rej, us)[0]] += 1.0 / N
            marg[x] += q[0, x] * acc
            marg += q[0, x] * (1 - acc) * res
        _close(marg, p[0], atol=4.0 / N)


def test_bonus_distribution(m):
    g = torch.Generator().manual_seed(5)
    p = _rand_dist(g, 2, 4)
    q = p[:1].clone()  # q == p 时一定接受
    N = 500
    grid = ((torch.arange(N, dtype=F64) + 0.5) / N).tolist()
    counts = torch.zeros(4, dtype=F64)
    for us in grid:
        out = m.verify(p, q, _L([2]), _t([0.999]), us)
        assert out[0] == 2 and len(out) == 2
        counts[out[1]] += 1.0 / N
    _close(counts, p[1], atol=2.0 / N)


def test_expected_accept_length_monte_carlo(m):
    g = torch.Generator().manual_seed(12)
    gamma, V = 3, 6
    p = _rand_dist(g, gamma + 1, V)
    q = _rand_dist(g, gamma, V)
    alpha = torch.minimum(p[:gamma], q).sum(dim=1)  # 每个位置的接受率，这里约 [0.59, 0.70, 0.88]
    expected = sum(torch.prod(alpha[:k]).item() for k in range(1, gamma + 1))
    T = 4000
    drafts = torch.stack([torch.multinomial(q[i], T, replacement=True, generator=g) for i in range(gamma)], dim=1)
    ua = torch.rand(T, gamma, generator=g, dtype=F64)
    us = torch.rand(T, generator=g, dtype=F64).tolist()
    total = 0
    for t in range(T):
        out = m.verify(p, q, drafts[t], ua[t], us[t])
        assert 1 <= len(out) <= gamma + 1
        total += len(out) - 1
    assert abs(total / T - expected) < 0.06
