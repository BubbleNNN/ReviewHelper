import numpy as np


def _bufs(p, n, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=n) for _ in range(p)]


def test_sum_on_every_rank(m):
    for p, n in [(2, 6), (4, 10), (5, 23), (8, 8)]:
        bufs = _bufs(p, n, p * 100 + n)
        results, _ = m.ring_allreduce(bufs)
        want = np.sum(bufs, axis=0)
        assert len(results) == p
        for r in range(p):
            np.testing.assert_allclose(results[r], want, atol=1e-10)


def test_step_count_and_neighbor_only(m):
    p = 6
    _, log = m.ring_allreduce(_bufs(p, 18, 1))
    steps = sorted({e[0] for e in log})
    assert steps == list(range(2 * (p - 1))), "应当恰好有 2(p-1) 步，编号连续"
    for step in steps:
        sends = [e for e in log if e[0] == step]
        assert len(sends) == p, "每一步每个 rank 恰好发一块"
        assert sorted(e[1] for e in sends) == list(range(p))
        for _, src, dst, _ in sends:
            assert dst == (src + 1) % p, "只能发给环上的下一个 rank"
        # 同一步里各 rank 发送的块号互不相同（链路满载、无重复）
        assert sorted(e[3] for e in sends) == list(range(p))


def test_bytes_per_rank(m):
    p, n = 4, 12  # 每块 3 个元素
    _, log = m.ring_allreduce(_bufs(p, n, 2))
    for r in range(p):
        sent = [e for e in log if e[1] == r]
        assert len(sent) == 2 * (p - 1)
        elems = len(sent) * (n // p)
        assert elems == 2 * (p - 1) / p * n  # = 18


def test_single_rank(m):
    x = np.arange(5.0)
    results, log = m.ring_allreduce([x])
    np.testing.assert_allclose(results[0], x)
    assert log == []


def test_input_not_modified(m):
    bufs = _bufs(3, 7, 3)
    copies = [b.copy() for b in bufs]
    m.ring_allreduce(bufs)
    for a, b in zip(bufs, copies):
        np.testing.assert_array_equal(a, b)


def test_integer_inputs(m):
    bufs = [np.full(7, r + 1) for r in range(4)]
    results, _ = m.ring_allreduce(bufs)
    for res in results:
        np.testing.assert_allclose(res, np.full(7, 10.0))


def test_alpha_beta_cost(m):
    # p=4: 2*3=6 次延迟；带宽项 2*3/4*1e9 字节 * 1e-10 s/B = 0.15 s
    t = m.allreduce_time(1e9, 4, alpha=1e-5, beta=1e-10)
    assert abs(t - (6e-5 + 0.15)) < 1e-12
    assert m.allreduce_time(1e9, 1, 1e-5, 1e-10) == 0
