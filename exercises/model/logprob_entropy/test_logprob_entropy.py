import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _randn(g, *shape):
    return torch.randn(*shape, generator=g, dtype=F64)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _brute_logprob(row, y):
    """纯 Python：row 是 python list，y 是 int。"""
    mx = max(row)
    return row[y] - (mx + math.log(math.fsum(math.exp(v - mx) for v in row)))


def _brute_entropy(row):
    mx = max(row)
    z = math.fsum(math.exp(v - mx) for v in row)
    ps = [math.exp(v - mx) / z for v in row]
    return -math.fsum(p * math.log(p) for p in ps if p > 0)


def test_logprobs_brute_force(m):
    g = _gen(0)
    logits = _randn(g, 2, 4, 7) * 3
    labels = torch.randint(0, 7, (2, 4), generator=g)  # long
    lp = m.logprobs_from_logits(logits, labels)
    assert lp.shape == (2, 4)
    for b in range(2):
        for t in range(4):
            ref = _brute_logprob(logits[b, t].tolist(), int(labels[b, t]))
            _close(lp[b, t], _t(ref), atol=1e-12)
    assert (lp <= 0).all()


def test_logprobs_large_logits(m):
    logits = _t([[[1000.0, 999.0, -1000.0]]])
    lp = m.logprobs_from_logits(logits, torch.tensor([[1]]))
    _close(lp[0, 0], _t(-1.0 - math.log(1 + math.exp(-1.0))), atol=1e-12)
    lp2 = m.logprobs_from_logits(logits, torch.tensor([[2]]))
    assert torch.isfinite(lp2[0, 0]) and lp2[0, 0].item() < -1999


def test_entropy_brute_force_and_bounds(m):
    logits = _randn(_gen(1), 3, 5, 6) * 2
    ent = m.entropy_from_logits(logits)
    assert ent.shape == (3, 5)
    for b in range(3):
        for t in range(5):
            _close(ent[b, t], _t(_brute_entropy(logits[b, t].tolist())), atol=1e-12)
    assert (ent >= -1e-12).all() and (ent <= math.log(6) + 1e-12).all()


def test_entropy_extremes(m):
    uniform = torch.zeros(1, 1, 10, dtype=F64)
    _close(m.entropy_from_logits(uniform)[0, 0], _t(math.log(10)), atol=1e-12)
    peaked = _t([[[50.0, 0.0, 0.0, 0.0]]])
    assert m.entropy_from_logits(peaked)[0, 0].item() < 1e-18 + 4 * 51 * math.exp(-50)


def test_response_alignment(m):
    # 序列长 6，response 长 3（位置 3,4,5）。构造 logits：位置 t 的 logits 在 input_ids[t+1] 处是 5，其余 0
    V = 8
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 0, 2, 2, 1, 3]])
    logits = torch.zeros(2, 6, V, dtype=F64)
    for b in range(2):
        for t in range(5):
            logits[b, t, input_ids[b, t + 1]] = 5.0
    lp = m.response_logprobs(logits, input_ids, 3)
    assert lp.shape == (2, 3)
    expected = 5.0 - math.log(math.exp(5.0) + V - 1)
    _close(lp, torch.full((2, 3), expected, dtype=F64), atol=1e-12)
    # 错位一格的话会取到 0 的那一列，值完全不同
    assert not torch.allclose(lp, torch.full_like(lp, -math.log(math.exp(5.0) + V - 1)))


def test_response_matches_manual_slice(m):
    g = _gen(2)
    logits = _randn(g, 2, 9, 11)
    ids = torch.randint(0, 11, (2, 9), generator=g)
    lp = m.response_logprobs(logits, ids, 4)
    for b in range(2):
        for j, t in enumerate(range(5, 9)):  # response token 位置 5..8，用位置 4..7 的 logits
            ref = _brute_logprob(logits[b, t - 1].tolist(), int(ids[b, t]))
            _close(lp[b, j], _t(ref), atol=1e-12)


def test_masked_mean(m):
    x = _t([[1.0, 2.0, 100.0], [3.0, 1e9, 1e9]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    _close(m.masked_mean(x, mask), _t(2.0), atol=1e-12)  # 0-d tensor


def test_masked_mean_keeps_grad(m):
    # masked_mean 要当 loss 用：返回 0-d tensor，梯度只落在 mask=True 的位置，大小 1/有效个数
    x = _randn(_gen(3), 2, 5).requires_grad_(True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 1, 1, 0]], dtype=torch.bool)
    out = m.masked_mean(x, mask)
    assert torch.is_tensor(out) and out.dim() == 0, "masked_mean 应返回 0-d tensor（不要 .item()）"
    out.backward()
    _close(x.grad, mask.to(F64) / 6.0, atol=1e-15)


def test_matches_torch_oracles(m):
    # torch 自带的实现当对照（题目里不许直接调用）
    g = _gen(4)
    B, T, V = 3, 7, 13
    logits = _randn(g, B, T, V) * 5
    ids = torch.randint(0, V, (B, T), generator=g)
    dist = Categorical(logits=logits)
    _close(m.logprobs_from_logits(logits, ids), dist.log_prob(ids), atol=1e-12)
    ce = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1), reduction="none").reshape(B, T)
    _close(m.logprobs_from_logits(logits, ids), -ce, atol=1e-12)
    _close(m.entropy_from_logits(logits), dist.entropy(), atol=1e-12)
    # HF 风格的对齐：先整体错一位（logits[:, :-1] 对 ids[:, 1:]），再取最后 R 个
    R = 4
    shifted = -F.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")  # (B, T-1)
    _close(m.response_logprobs(logits, ids, R), shifted[:, -R:], atol=1e-12)
    # float32 输入应得到 float32 输出
    l32 = logits.float()
    _close(m.logprobs_from_logits(l32, ids), Categorical(logits=l32).log_prob(ids), atol=1e-4)
    _close(m.entropy_from_logits(l32), Categorical(logits=l32).entropy(), atol=1e-4)
