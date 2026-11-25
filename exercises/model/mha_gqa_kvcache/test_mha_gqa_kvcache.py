import math

import numpy as np


def _params(rng, d_model, n_heads, n_kv, hd):
    s = 1.0 / math.sqrt(d_model)
    return {
        "wq": rng.normal(size=(d_model, n_heads * hd)) * s,
        "wk": rng.normal(size=(d_model, n_kv * hd)) * s,
        "wv": rng.normal(size=(d_model, n_kv * hd)) * s,
        "wo": rng.normal(size=(n_heads * hd, d_model)) * s,
    }


def _loop_reference(x, p, n_heads, n_kv, causal=True):
    """逐 batch、逐 head、逐位置的循环实现：head h 用 kv head h // (n_heads // n_kv)。"""
    B, T, D = x.shape
    hd = p["wq"].shape[1] // n_heads
    group = n_heads // n_kv
    out_heads = np.zeros((B, T, n_heads * hd))
    for b in range(B):
        Q = x[b] @ p["wq"]
        K = x[b] @ p["wk"]
        V = x[b] @ p["wv"]
        for h in range(n_heads):
            g = h // group
            qh = Q[:, h * hd:(h + 1) * hd]
            kh = K[:, g * hd:(g + 1) * hd]
            vh = V[:, g * hd:(g + 1) * hd]
            for i in range(T):
                js = range(i + 1) if causal else range(T)
                s = np.array([qh[i] @ kh[j] / math.sqrt(hd) for j in js])
                w = np.exp(s - s.max())
                w /= w.sum()
                out_heads[b, i, h * hd:(h + 1) * hd] = sum(w[t] * vh[j] for t, j in enumerate(js))
    return out_heads @ p["wo"]


def test_repeat_kv_mapping(m):
    x = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    y = m.repeat_kv(x, 2)
    assert y.shape == (2, 6, 4, 5)
    for h in range(6):
        np.testing.assert_array_equal(y[:, h], x[:, h // 2])
    np.testing.assert_array_equal(m.repeat_kv(x, 1), x)


def test_mha_matches_loop(m):
    rng = np.random.default_rng(0)
    p = _params(rng, 12, 3, 3, 4)
    x = rng.normal(size=(2, 5, 12))
    np.testing.assert_allclose(m.gqa_attention(x, p, 3, 3), _loop_reference(x, p, 3, 3), atol=1e-10)
    np.testing.assert_allclose(m.gqa_attention(x, p, 3, 3, causal=False),
                               _loop_reference(x, p, 3, 3, causal=False), atol=1e-10)


def test_gqa_and_mqa_match_loop(m):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 6, 16))
    p = _params(rng, 16, 4, 2, 4)  # GQA：每 2 个 query head 共享 1 个 kv head
    np.testing.assert_allclose(m.gqa_attention(x, p, 4, 2), _loop_reference(x, p, 4, 2), atol=1e-10)
    p1 = _params(rng, 16, 4, 1, 4)  # MQA
    np.testing.assert_allclose(m.gqa_attention(x, p1, 4, 1), _loop_reference(x, p1, 4, 1), atol=1e-10)


def test_gqa_equals_mha_with_duplicated_kv_weights(m):
    rng = np.random.default_rng(2)
    n_heads, n_kv, hd, D = 4, 2, 3, 10
    p = _params(rng, D, n_heads, n_kv, hd)
    # 把每个 kv head 的权重按 repeat_kv 约定复制成 n_heads 份，得到等价的 MHA
    cols_k = [p["wk"][:, (h // 2) * hd:(h // 2 + 1) * hd] for h in range(n_heads)]
    cols_v = [p["wv"][:, (h // 2) * hd:(h // 2 + 1) * hd] for h in range(n_heads)]
    p_mha = dict(p, wk=np.concatenate(cols_k, axis=1), wv=np.concatenate(cols_v, axis=1))
    x = rng.normal(size=(1, 7, D))
    np.testing.assert_allclose(m.gqa_attention(x, p, n_heads, n_kv),
                               m.gqa_attention(x, p_mha, n_heads, n_heads), atol=1e-10)


def test_decode_with_cache_equals_full_forward(m):
    rng = np.random.default_rng(3)
    p = _params(rng, 16, 4, 2, 4)
    x = rng.normal(size=(2, 7, 16))
    full = m.gqa_attention(x, p, 4, 2, causal=True)
    cache = m.KVCache()
    assert len(cache) == 0
    steps = [m.gqa_decode_step(x[:, t:t + 1], p, 4, 2, cache) for t in range(7)]
    np.testing.assert_allclose(np.concatenate(steps, axis=1), full, atol=1e-10)
    assert len(cache) == 7
    assert cache.k.shape == (2, 2, 7, 4), "cache 只存 n_kv_heads 个 head"


def test_prefill_chunk_then_decode(m):
    rng = np.random.default_rng(4)
    p = _params(rng, 8, 2, 1, 4)
    x = rng.normal(size=(1, 6, 8))
    full = m.gqa_attention(x, p, 2, 1)
    cache = m.KVCache()
    a = m.gqa_decode_step(x[:, :4], p, 2, 1, cache)  # prefill 4 个 token
    b = m.gqa_decode_step(x[:, 4:5], p, 2, 1, cache)
    c = m.gqa_decode_step(x[:, 5:6], p, 2, 1, cache)
    np.testing.assert_allclose(np.concatenate([a, b, c], axis=1), full, atol=1e-10)


def test_causality(m):
    rng = np.random.default_rng(5)
    p = _params(rng, 8, 2, 2, 4)
    x = rng.normal(size=(1, 6, 8))
    y = m.gqa_attention(x, p, 2, 2)
    x2 = x.copy()
    x2[:, 4:] = rng.normal(size=(1, 2, 8)) * 10
    y2 = m.gqa_attention(x2, p, 2, 2)
    np.testing.assert_allclose(y[:, :4], y2[:, :4], atol=1e-12)
    assert not np.allclose(y[:, 4:], y2[:, 4:])
