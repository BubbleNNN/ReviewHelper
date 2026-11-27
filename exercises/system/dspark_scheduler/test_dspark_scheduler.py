import itertools
import operator

import torch

F64 = torch.float64


def _theta(conf, lengths, sps):
    """测试方自己按定义算 Θ = τ · SPS(B)"""
    R = len(conf)
    tau, B = float(R), R
    for cs, l in zip(conf, lengths):
        prod = 1.0
        for c in cs[:l]:
            prod *= c
            tau += prod
        B += l
    return tau * sps(B)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _uniform(g, lo, hi, n):
    """n 个 [lo, hi) 上均匀分布的 python float"""
    return (lo + (hi - lo) * torch.rand(n, generator=g, dtype=F64)).tolist()


def _randint(g, lo, hi):
    """[lo, hi) 内的 python int"""
    return int(torch.randint(lo, hi, (), generator=g))


def _rand_conf(g, R, gamma):
    return [_uniform(g, 0.3, 0.98, gamma) for _ in range(R)]


def _approx(xs, ys, atol=1e-12):
    xs = list(xs)
    assert len(xs) == len(ys) and all(abs(x - y) <= atol for x, y in zip(xs, ys)), f"{xs} != {ys}"


def test_prefix_survival(m):
    a = m.prefix_survival([[0.9, 0.8, 0.5], [0.6], []])
    _approx(a[0], [0.9, 0.72, 0.36])
    _approx(a[1], [0.6])
    assert list(a[2]) == []


def test_flat_sps_verifies_all_positive(m):
    conf = [[0.9, 0.0, 0.8], [0.5, 0.5, 0.5]]
    lengths, th = m.greedy_schedule(conf, lambda B: 1.0)
    assert list(lengths) == [1, 3], "a=0 之后的位置不该验证"
    assert abs(th - 3.775) < 1e-12  # τ = 2 + 0.9 + (0.5+0.25+0.125)


def test_steep_sps_verifies_nothing(m):
    conf = [[0.95, 0.9], [0.99, 0.99, 0.99]]
    lengths, th = m.greedy_schedule(conf, lambda B: 1.0 if B <= 2 else 1e-3)
    assert list(lengths) == [0, 0]
    assert abs(th - 2.0) < 1e-12


def test_greedy_optimal_when_unimodal(m):
    g = _gen(0)
    for trial in range(12):
        R, gamma = _randint(g, 1, 4), _randint(g, 1, 4)
        conf = _rand_conf(g, R, gamma)
        (c0,), (c1,) = _uniform(g, 1, 20, 1), _uniform(g, 0.2, 3, 1)
        sps = lambda B, c0=c0, c1=c1: 1.0 / (c0 + c1 * B)  # noqa: E731
        lg, tg = m.greedy_schedule(conf, sps)
        lb, tb = m.brute_force_schedule(conf, sps)
        assert abs(tg - tb) < 1e-12, f"trial {trial}: greedy={tg} brute={tb}"
        assert list(lg) == list(lb)
        assert abs(tg - _theta(conf, lg, sps)) < 1e-12


def test_admitted_set_is_top_k(m):
    conf = _rand_conf(_gen(1), 6, 5)
    lengths, _ = m.greedy_schedule(conf, lambda B: 1.0 / (8 + B))
    a = [list(itertools.accumulate(c, operator.mul)) for c in conf]
    inside = [a[r][j] for r in range(6) for j in range(lengths[r])]
    outside = [a[r][j] for r in range(6) for j in range(lengths[r], 5)]
    assert inside and outside
    assert min(inside) >= max(outside), "被验证的应是全局前缀存活概率最高的那些位置"
    for r in range(6):
        assert 0 <= lengths[r] <= 5


def test_jagged_sps_breaks_early_stop(m):
    # 论文的提醒：早停只有在 Θ 单峰时才是全局最优；真实 SPS 有台阶
    conf = [[0.9, 0.9, 0.9, 0.9]]
    sps = lambda B: 1.0 if B <= 1 else 0.5  # noqa: E731
    lg, tg = m.greedy_schedule(conf, sps)
    lb, tb = m.brute_force_schedule(conf, sps)
    assert list(lg) == [0] and abs(tg - 1.0) < 1e-12
    assert list(lb) == [4] and abs(tb - 0.5 * (1 + 0.9 + 0.81 + 0.729 + 0.6561)) < 1e-12
    assert tb > tg


def test_brute_force_matches_definition(m):
    g = _gen(2)
    conf = [_uniform(g, 0.2, 0.95, n) for n in (2, 0, 3)]
    sps = lambda B: 1.0 / (1 + 0.4 * B)  # noqa: E731
    lb, tb = m.brute_force_schedule(conf, sps)
    best = max(_theta(conf, (x, 0, z), sps) for x in range(3) for z in range(4))
    assert abs(tb - best) < 1e-12
    assert abs(_theta(conf, lb, sps) - best) < 1e-12
