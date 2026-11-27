import math

import torch

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _close(actual, expected, atol=1e-12):
    torch.testing.assert_close(actual, _t(expected) if not torch.is_tensor(expected) else expected, atol=atol, rtol=0)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def test_top_k_hand(m):
    p = _t([0.1, 0.4, 0.3, 0.2])
    _close(m.top_k_filter(p, 2), [0, 4 / 7, 3 / 7, 0])
    _close(m.top_k_filter(p, 1), [0, 1, 0, 0])
    _close(m.top_k_filter(p, 0), p)
    _close(m.top_k_filter(p, 10), p)


def test_top_k_ties_kept(m):
    p = _t([0.4, 0.2, 0.2, 0.2])
    _close(m.top_k_filter(p, 2), p)  # 第 2 名有三个并列，全保留


def test_top_p_minimal_prefix(m):
    p = _t([0.15, 0.5, 0.05, 0.3])  # 降序：0.5, 0.3, 0.15, 0.05；累计 0.5, 0.8, 0.95, 1.0
    _close(m.top_p_filter(p, 0.75), [0, 0.625, 0, 0.375])
    _close(m.top_p_filter(p, 0.81), [0.15 / 0.95, 0.5 / 0.95, 0, 0.3 / 0.95])
    _close(m.top_p_filter(p, 0.3), [0, 1, 0, 0])  # 至少保留 1 个
    _close(m.top_p_filter(p, 1.0), p)


def test_top_p_stable_tie_break(m):
    p = _t([0.2, 0.4, 0.2, 0.2])
    _close(m.top_p_filter(p, 0.5), [1 / 3, 2 / 3, 0, 0])


def test_temperature_and_order(m):
    logits = _t([1.0, 2.0, 3.0])
    z = [math.exp(x / 2.0) for x in (1.0, 2.0, 3.0)]
    _close(m.process_logits(logits, temperature=2.0), [x / sum(z) for x in z])
    # greedy
    _close(m.process_logits(_t([0.5, 3.0, 3.0, 1.0]), temperature=0.0), [0, 1, 0, 0], atol=0)
    sharp = m.process_logits(logits, temperature=1e-3)
    assert sharp[2] > 1 - 1e-12
    # 先 top-k 再 top-p：probs∝e^[0,1,2,3]，top_k=3 → [e,e²,e³]/Σ，再 top_p=0.7
    lg = _t([0.0, 1.0, 2.0, 3.0])
    e = torch.exp(_t([1.0, 2.0, 3.0]))
    pk = e / e.sum()  # ≈ [0.090, 0.245, 0.665]
    assert pk[2] < 0.7 < pk[2] + pk[1]
    expect = torch.stack([pk.new_zeros(()), pk.new_zeros(()), pk[1], pk[2]]) / (pk[1] + pk[2])
    _close(m.process_logits(lg, top_k=3, top_p=0.7), expect)


def test_sampler_frequencies(m):
    g = _gen(0)
    logits = _t([0.0, 1.0, 2.0, -1.0])
    n = 4000
    counts = torch.zeros(4, dtype=F64)
    for _ in range(n):
        counts[m.sample(logits, g, top_k=3)] += 1
    e = torch.exp(_t([0.0, 1.0, 2.0]))
    target = torch.cat([e / e.sum(), _t([0.0])])
    assert counts[3] == 0, "top_k=3 时第 4 个 token 永远不该被采到"
    _close(counts / n, target, atol=0.035)


def test_sampler_uses_given_generator_and_greedy(m):
    logits = _t([0.3, -1.0, 2.5, 0.0])
    a = [m.sample(logits, _gen(7)) for _ in range(5)]
    b = [m.sample(logits, _gen(7)) for _ in range(5)]
    assert a == b, "同一个种子应得到同样的结果"
    g = _gen(1)
    assert all(m.sample(logits, g, temperature=0.0) == 2 for _ in range(20))
    assert all(m.sample(logits, g, top_k=1) == 2 for _ in range(20))
    assert isinstance(m.sample(logits, g), int), "sample 应返回 python int（用 .item()）"
