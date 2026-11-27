import torch

F64 = torch.float64


def _bufs(p, n, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, generator=g, dtype=F64) for _ in range(p)]


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def test_sum_on_every_rank(m):
    for p, n in [(2, 6), (4, 10), (5, 23), (8, 8)]:
        bufs = _bufs(p, n, p * 100 + n)
        results, _ = m.ring_allreduce(bufs)
        want = torch.stack(bufs).sum(dim=0)  # oracle：直接逐元素求和
        assert len(results) == p
        for r in range(p):
            _close(results[r], want, atol=1e-10)


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
    x = torch.arange(5, dtype=F64)
    results, log = m.ring_allreduce([x])
    _close(results[0], x, atol=0)
    assert log == []


def test_input_not_modified(m):
    bufs = _bufs(3, 7, 3)
    copies = [b.clone() for b in bufs]
    m.ring_allreduce(bufs)
    for a, b in zip(bufs, copies):
        assert torch.equal(a, b), "输入被原地修改了（tensor_split 返回的是 view，要先 clone）"


def test_integer_inputs(m):
    # int64 输入：结果保持 int64，且精确等于 1+2+3+4
    bufs = [torch.full((7,), r + 1, dtype=torch.long) for r in range(4)]
    results, _ = m.ring_allreduce(bufs)
    for res in results:
        _close(res, torch.full((7,), 10, dtype=torch.long), atol=0)


def test_alpha_beta_cost(m):
    # p=4: 2*3=6 次延迟；带宽项 2*3/4*1e9 字节 * 1e-10 s/B = 0.15 s
    t = m.allreduce_time(1e9, 4, alpha=1e-5, beta=1e-10)
    assert isinstance(t, float), "allreduce_time 应返回 python float"
    assert abs(t - (6e-5 + 0.15)) < 1e-12
    assert m.allreduce_time(1e9, 1, 1e-5, 1e-10) == 0
