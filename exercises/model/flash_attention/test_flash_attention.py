import numpy as np


def _naive(q, k, v, causal=False):
    s = q @ k.T / np.sqrt(q.shape[1])
    if causal:
        s = np.where(np.tril(np.ones(s.shape, dtype=bool)), s, -np.inf)
    mx = s.max(axis=1, keepdims=True)
    e = np.exp(s - mx)
    z = e.sum(axis=1, keepdims=True)
    return (e / z) @ v, (mx + np.log(z))[:, 0]


def _qkv(seed, tq, tk, d=8, dv=6, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(tq, d)) * scale, rng.normal(size=(tk, d)), rng.normal(size=(tk, dv))


def test_matches_naive(m):
    q, k, v = _qkv(0, 32, 32)
    out, lse = m.flash_attention(q, k, v, block_q=8, block_k=8)
    ro, rl = _naive(q, k, v)
    np.testing.assert_allclose(out, ro, atol=1e-10)
    np.testing.assert_allclose(lse, rl, atol=1e-10)


def test_causal(m):
    q, k, v = _qkv(1, 40, 40)
    out, lse = m.flash_attention(q, k, v, causal=True, block_q=16, block_k=8)
    ro, rl = _naive(q, k, v, causal=True)
    np.testing.assert_allclose(out, ro, atol=1e-10)
    np.testing.assert_allclose(lse, rl, atol=1e-10)
    np.testing.assert_allclose(out[0], v[0], atol=1e-12)  # 第一个 query 只能看第一个 key


def test_non_divisible_lengths(m):
    q, k, v = _qkv(2, 37, 23)
    out, lse = m.flash_attention(q, k, v, block_q=8, block_k=5)
    ro, rl = _naive(q, k, v)
    np.testing.assert_allclose(out, ro, atol=1e-10)
    np.testing.assert_allclose(lse, rl, atol=1e-10)
    q, k, v = _qkv(3, 37, 37)
    out, _ = m.flash_attention(q, k, v, causal=True, block_q=7, block_k=5)
    np.testing.assert_allclose(out, _naive(q, k, v, causal=True)[0], atol=1e-10)


def test_tiny_and_huge_blocks(m):
    q, k, v = _qkv(4, 9, 9)
    ro, rl = _naive(q, k, v, causal=True)
    for bq, bk in ((1, 1), (1, 4), (3, 1), (100, 100)):
        out, lse = m.flash_attention(q, k, v, causal=True, block_q=bq, block_k=bk)
        np.testing.assert_allclose(out, ro, atol=1e-10, err_msg=f"block=({bq},{bk})")
        np.testing.assert_allclose(lse, rl, atol=1e-10)


def test_large_scores_are_stable(m):
    # 分数量级上百，逐块的最大值会不断被刷新
    q, k, v = _qkv(5, 20, 30, scale=60.0)
    out, lse = m.flash_attention(q, k, v, block_q=4, block_k=4)
    assert np.all(np.isfinite(out)) and np.all(np.isfinite(lse))
    ro, rl = _naive(q, k, v)
    np.testing.assert_allclose(out, ro, atol=1e-9)
    np.testing.assert_allclose(lse, rl, rtol=1e-12)


def test_increasing_scores_order(m):
    # 构造越往后的 key 分数越大：每个新块都会把行最大值抬高，专门测 e^{m-m'} 的换算
    d = 4
    q = np.ones((3, d))
    k = np.arange(12, dtype=float)[:, None] * np.ones((1, d)) * 0.7
    v = np.arange(12, dtype=float)[:, None] * np.array([[1.0, -1.0]])
    out, lse = m.flash_attention(q, k, v, block_q=2, block_k=3)
    ro, rl = _naive(q, k, v)
    np.testing.assert_allclose(out, ro, atol=1e-10)
    np.testing.assert_allclose(lse, rl, atol=1e-10)
