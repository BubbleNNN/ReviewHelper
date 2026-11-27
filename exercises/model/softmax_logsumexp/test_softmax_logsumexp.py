import math

import torch

F64 = torch.float64
INF = math.inf


def _t(x):
    return torch.tensor(x, dtype=F64)


def _randn(g, *shape):
    return torch.randn(*shape, generator=g, dtype=F64)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _softmax_ref(row):
    """按定义直接算（纯 Python），只用于数值不大的输入。row 是 python list。"""
    e = [math.exp(float(v)) for v in row]
    s = math.fsum(e)
    return [x / s for x in e]


def test_small_values_match_definition(m):
    x = _t([1.0, 2.0, 3.0])
    _close(m.softmax(x), _t(_softmax_ref([1, 2, 3])), atol=1e-12)
    expected_lse = math.log(math.exp(1) + math.exp(2) + math.exp(3))
    _close(m.logsumexp(x), _t(expected_lse), atol=1e-12)  # 一维输入 -> 0-d tensor


def test_huge_logits_do_not_overflow(m):
    x = _t([1e4, 1e4 + 1.0, 1e4 - 1.0])
    p = m.softmax(x)
    assert torch.isfinite(p).all()
    _close(p, _t(_softmax_ref([0.0, 1.0, -1.0])), atol=1e-12)
    expected = 1e4 + math.log(math.exp(0.0) + math.exp(1.0) + math.exp(-1.0))
    _close(m.logsumexp(x), _t(expected), atol=1e-9)  # 量级 1e4，float64 的分辨率约 2e-12


def test_very_negative_logits_do_not_underflow(m):
    x = _t([-1000.0, -1000.0])
    _close(m.logsumexp(x), _t(-1000.0 + math.log(2.0)), atol=1e-10)
    _close(m.softmax(x), _t([0.5, 0.5]), atol=1e-12)
    _close(m.log_softmax(x), _t([-math.log(2.0)] * 2), atol=1e-12)


def test_shift_invariance(m):
    x = _randn(_gen(0), 4, 7)
    _close(m.softmax(x), m.softmax(x + 123.0), atol=1e-12)
    _close(m.logsumexp(x + 5.0), m.logsumexp(x) + 5.0, atol=1e-12)


def test_neg_inf_mask(m):
    x = _t([0.0, -INF, 0.0, -INF])
    _close(m.softmax(x), _t([0.5, 0.0, 0.5, 0.0]), atol=1e-15)
    ls = m.log_softmax(x)
    assert ls[1].item() == -INF and ls[3].item() == -INF
    _close(ls[[0, 2]], _t([-math.log(2.0)] * 2), atol=1e-12)
    out = m.logsumexp(_t([[-INF, -INF], [0.0, -INF]]))
    assert out[0].item() == -INF, "整行 -inf 时 logsumexp 应为 -inf，而不是 nan"
    _close(out[1], _t(0.0), atol=1e-15)


def test_axis_and_keepdims(m):
    x = _randn(_gen(1), 3, 5)
    out0 = m.logsumexp(x, dim=0)
    assert out0.shape == (5,)
    for j in range(5):
        ref = math.log(math.fsum(math.exp(v) for v in x[:, j].tolist()))
        _close(out0[j], _t(ref), atol=1e-12)
    assert m.logsumexp(x, dim=1, keepdim=True).shape == (3, 1)
    p = m.softmax(x, dim=0)
    assert p.shape == x.shape
    _close(p.sum(dim=0), torch.ones(5, dtype=F64), atol=1e-12)


def test_log_softmax_consistent_with_softmax(m):
    x = _randn(_gen(2), 2, 6) * 3
    ls = m.log_softmax(x)
    for b in range(2):
        _close(torch.exp(ls[b]), _t(_softmax_ref(x[b].tolist())), atol=1e-12)
    # 概率极小（e^-5000）的位置：log(softmax) 会变成 -inf，正确实现应给出有限值
    big = _t([[5e3, 0.0, -5e3]])
    lb = m.log_softmax(big)
    assert torch.isfinite(lb).all()
    _close(lb[0], _t([0.0, -5e3, -1e4]), atol=1e-9)


def test_matches_torch_builtins(m):
    # torch 自带的实现当对照（题目里不许直接调用它们）
    g = _gen(3)
    x = _randn(g, 3, 4, 5) * 4
    x[0, 1, 2] = -INF  # 零散的 mask 位置；沿任何一维都没有整行 -inf（那时 softmax 无定义）
    x[2, 3, 0] = -INF
    x[1, 0, 4] = -INF
    for dim in (0, 1, -1):
        _close(m.softmax(x, dim=dim), torch.softmax(x, dim=dim), atol=1e-12)
        _close(m.log_softmax(x, dim=dim), torch.log_softmax(x, dim=dim), atol=1e-12)
        _close(m.logsumexp(x, dim=dim), torch.logsumexp(x, dim=dim), atol=1e-12)
        _close(m.logsumexp(x, dim=dim, keepdim=True), torch.logsumexp(x, dim=dim, keepdim=True), atol=1e-12)
    full_masked = _t([[-INF, -INF, -INF], [1.0, -INF, 2.0]])
    _close(m.logsumexp(full_masked), torch.logsumexp(full_masked, dim=-1), atol=1e-12)
    # float32 输入应得到 float32 输出（不要偷偷升精度）
    x32 = torch.randn(4, 6, generator=g) * 4
    _close(m.softmax(x32), torch.softmax(x32, dim=-1), atol=1e-6)
    _close(m.log_softmax(x32), torch.log_softmax(x32, dim=-1), atol=1e-5)
    _close(m.logsumexp(x32), torch.logsumexp(x32, dim=-1), atol=1e-5)
