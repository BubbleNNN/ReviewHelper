import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def _close(actual, expected, atol, msg=None):
    kw = {} if msg is None else {"msg": lambda s: f"{msg}\n{s}"}
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0, **kw)


def _naive(q, k, v, causal=False):
    """参照：一次性算完整的 softmax(QK^T / sqrt(d)) V 和每行的 logsumexp"""
    s = q @ k.T / math.sqrt(q.shape[1])
    if causal:
        keep = torch.tril(torch.ones(s.shape, dtype=torch.bool))
        s = s.masked_fill(~keep, -math.inf)
    return torch.softmax(s, dim=1) @ v, torch.logsumexp(s, dim=1)


def _qkv(seed, tq, tk, d=8, dv=6, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(tq, d, generator=g, dtype=F64) * scale
    k = torch.randn(tk, d, generator=g, dtype=F64)
    v = torch.randn(tk, dv, generator=g, dtype=F64)
    return q, k, v


def test_matches_naive(m):
    q, k, v = _qkv(0, 32, 32)
    out, lse = m.flash_attention(q, k, v, block_q=8, block_k=8)
    ro, rl = _naive(q, k, v)
    _close(out, ro, atol=1e-10)
    _close(lse, rl, atol=1e-10)


def test_causal(m):
    q, k, v = _qkv(1, 40, 40)
    out, lse = m.flash_attention(q, k, v, causal=True, block_q=16, block_k=8)
    ro, rl = _naive(q, k, v, causal=True)
    _close(out, ro, atol=1e-10)
    _close(lse, rl, atol=1e-10)
    _close(out[0], v[0], atol=1e-12)  # 第一个 query 只能看第一个 key


def test_non_divisible_lengths(m):
    q, k, v = _qkv(2, 37, 23)
    out, lse = m.flash_attention(q, k, v, block_q=8, block_k=5)
    ro, rl = _naive(q, k, v)
    _close(out, ro, atol=1e-10)
    _close(lse, rl, atol=1e-10)
    q, k, v = _qkv(3, 37, 37)
    out, _ = m.flash_attention(q, k, v, causal=True, block_q=7, block_k=5)
    _close(out, _naive(q, k, v, causal=True)[0], atol=1e-10)


def test_tiny_and_huge_blocks(m):
    q, k, v = _qkv(4, 9, 9)
    ro, rl = _naive(q, k, v, causal=True)
    for bq, bk in ((1, 1), (1, 4), (3, 1), (100, 100)):
        out, lse = m.flash_attention(q, k, v, causal=True, block_q=bq, block_k=bk)
        _close(out, ro, atol=1e-10, msg=f"block=({bq},{bk})")
        _close(lse, rl, atol=1e-10, msg=f"block=({bq},{bk})")


def test_large_scores_are_stable(m):
    # 分数量级上百，逐块的最大值会不断被刷新
    q, k, v = _qkv(5, 20, 30, scale=60.0)
    out, lse = m.flash_attention(q, k, v, block_q=4, block_k=4)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    ro, rl = _naive(q, k, v)
    _close(out, ro, atol=1e-9)
    _close(lse, rl, atol=1e-9)


def test_increasing_scores_order(m):
    # 构造越往后的 key 分数越大：每个新块都会把行最大值抬高，专门测 e^{m-m'} 的换算
    d = 4
    q = torch.ones(3, d, dtype=F64)
    k = torch.arange(12, dtype=F64)[:, None] * torch.ones(1, d, dtype=F64) * 0.7
    v = torch.arange(12, dtype=F64)[:, None] * torch.tensor([[1.0, -1.0]], dtype=F64)
    out, lse = m.flash_attention(q, k, v, block_q=2, block_k=3)
    ro, rl = _naive(q, k, v)
    _close(out, ro, atol=1e-10)
    _close(lse, rl, atol=1e-10)


def test_matches_sdpa(m):
    # 参照：PyTorch 自带的 F.scaled_dot_product_attention（只比较 out；lse 用完整分数矩阵的 logsumexp）
    cases = [
        (False, 32, 32, 8, 8),
        (False, 19, 41, 6, 10),  # Tq != Tk，且都不能整除
        (False, 25, 13, 32, 4),
        (True, 33, 33, 16, 5),
        (True, 50, 50, 7, 3),
        (True, 17, 17, 4, 32),
    ]
    for i, (causal, tq, tk, bq, bk) in enumerate(cases):
        q, k, v = _qkv(10 + i, tq, tk, d=16, dv=5)
        out, lse = m.flash_attention(q, k, v, causal=causal, block_q=bq, block_k=bk)
        tag = f"causal={causal}, T=({tq},{tk}), block=({bq},{bk})"
        _close(out, F.scaled_dot_product_attention(q, k, v, is_causal=causal), atol=1e-10, msg=tag)
        s = q @ k.T / math.sqrt(q.shape[1])
        if causal:
            s = s.masked_fill(torch.ones(tq, tk, dtype=torch.bool).triu(1), -math.inf)
        _close(lse, torch.logsumexp(s, dim=1), atol=1e-10, msg=tag)
    # float32 输入，输出也应是 float32
    q, k, v = (t.float() for t in _qkv(20, 24, 24))
    out, lse = m.flash_attention(q, k, v, causal=True, block_q=8, block_k=8)
    assert out.dtype == torch.float32 and lse.dtype == torch.float32, "输出 dtype 应与输入一致"
    _close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5)
