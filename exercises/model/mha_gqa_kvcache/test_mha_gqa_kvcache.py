import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _randn(g, *shape):
    return torch.randn(*shape, generator=g, dtype=F64)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _params(g, d_model, n_heads, n_kv, hd):
    s = 1.0 / math.sqrt(d_model)
    return {
        "wq": _randn(g, d_model, n_heads * hd) * s,
        "wk": _randn(g, d_model, n_kv * hd) * s,
        "wv": _randn(g, d_model, n_kv * hd) * s,
        "wo": _randn(g, n_heads * hd, d_model) * s,
    }


def _loop_reference(x, p, n_heads, n_kv, causal=True):
    """逐 batch、逐 head、逐位置的循环实现：head h 用 kv head h // (n_heads // n_kv)。"""
    B, T, D = x.shape
    hd = p["wq"].shape[1] // n_heads
    group = n_heads // n_kv
    out_heads = torch.zeros(B, T, n_heads * hd, dtype=x.dtype)
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
                s = torch.stack([qh[i] @ kh[j] / math.sqrt(hd) for j in js])
                w = torch.exp(s - s.max())
                w = w / w.sum()
                out_heads[b, i, h * hd:(h + 1) * hd] = sum(w[t] * vh[j] for t, j in enumerate(js))
    return out_heads @ p["wo"]


def _sdpa_reference(x, p, n_heads, n_kv, causal=True):
    """用 torch 自带的 F.scaled_dot_product_attention(enable_gqa=True) 当对照（它内部按 h // n_rep 共享 KV）。"""
    B, T, _ = x.shape
    hd = p["wq"].shape[1] // n_heads
    q = (x @ p["wq"]).view(B, T, n_heads, hd).transpose(1, 2)
    k = (x @ p["wk"]).view(B, T, n_kv, hd).transpose(1, 2)
    v = (x @ p["wv"]).view(B, T, n_kv, hd).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
    return o.transpose(1, 2).reshape(B, T, n_heads * hd) @ p["wo"]


def test_repeat_kv_mapping(m):
    x = torch.arange(2 * 3 * 4 * 5, dtype=F64).reshape(2, 3, 4, 5)
    y = m.repeat_kv(x, 2)
    assert y.shape == (2, 6, 4, 5)
    for h in range(6):
        _close(y[:, h], x[:, h // 2], atol=0)
    _close(m.repeat_kv(x, 1), x, atol=0)


def test_mha_matches_loop(m):
    g = _gen(0)
    p = _params(g, 12, 3, 3, 4)
    x = _randn(g, 2, 5, 12)
    _close(m.gqa_attention(x, p, 3, 3), _loop_reference(x, p, 3, 3), atol=1e-10)
    _close(m.gqa_attention(x, p, 3, 3, causal=False), _loop_reference(x, p, 3, 3, causal=False), atol=1e-10)


def test_gqa_and_mqa_match_loop(m):
    g = _gen(1)
    x = _randn(g, 2, 6, 16)
    p = _params(g, 16, 4, 2, 4)  # GQA：每 2 个 query head 共享 1 个 kv head
    _close(m.gqa_attention(x, p, 4, 2), _loop_reference(x, p, 4, 2), atol=1e-10)
    p1 = _params(g, 16, 4, 1, 4)  # MQA
    _close(m.gqa_attention(x, p1, 4, 1), _loop_reference(x, p1, 4, 1), atol=1e-10)


def test_gqa_equals_mha_with_duplicated_kv_weights(m):
    g = _gen(2)
    n_heads, n_kv, hd, D = 4, 2, 3, 10
    p = _params(g, D, n_heads, n_kv, hd)
    # 把每个 kv head 的权重按 repeat_kv 约定复制成 n_heads 份，得到等价的 MHA
    cols_k = [p["wk"][:, (h // 2) * hd:(h // 2 + 1) * hd] for h in range(n_heads)]
    cols_v = [p["wv"][:, (h // 2) * hd:(h // 2 + 1) * hd] for h in range(n_heads)]
    p_mha = dict(p, wk=torch.cat(cols_k, dim=1), wv=torch.cat(cols_v, dim=1))
    x = _randn(g, 1, 7, D)
    _close(m.gqa_attention(x, p, n_heads, n_kv), m.gqa_attention(x, p_mha, n_heads, n_heads), atol=1e-10)


def test_decode_with_cache_equals_full_forward(m):
    g = _gen(3)
    p = _params(g, 16, 4, 2, 4)
    x = _randn(g, 2, 7, 16)
    full = m.gqa_attention(x, p, 4, 2, causal=True)
    cache = m.KVCache()
    assert len(cache) == 0
    steps = [m.gqa_decode_step(x[:, t:t + 1], p, 4, 2, cache) for t in range(7)]
    _close(torch.cat(steps, dim=1), full, atol=1e-10)
    assert len(cache) == 7
    assert cache.k.shape == (2, 2, 7, 4), "cache 只存 n_kv_heads 个 head"


def test_prefill_chunk_then_decode(m):
    g = _gen(4)
    p = _params(g, 8, 2, 1, 4)
    x = _randn(g, 1, 6, 8)
    full = m.gqa_attention(x, p, 2, 1)
    cache = m.KVCache()
    a = m.gqa_decode_step(x[:, :4], p, 2, 1, cache)  # prefill 4 个 token
    b = m.gqa_decode_step(x[:, 4:5], p, 2, 1, cache)
    c = m.gqa_decode_step(x[:, 5:6], p, 2, 1, cache)
    _close(torch.cat([a, b, c], dim=1), full, atol=1e-10)


def test_causality(m):
    g = _gen(5)
    p = _params(g, 8, 2, 2, 4)
    x = _randn(g, 1, 6, 8)
    y = m.gqa_attention(x, p, 2, 2)
    x2 = x.clone()
    x2[:, 4:] = _randn(g, 1, 2, 8) * 10
    y2 = m.gqa_attention(x2, p, 2, 2)
    _close(y[:, :4], y2[:, :4], atol=1e-12)
    assert not torch.allclose(y[:, 4:], y2[:, 4:])


def test_multi_token_chunk_after_cache(m):
    # cache 非空时再喂一段 t>1 的 chunk（分块 prefill / 投机解码验证）：mask 必须右下角对齐
    g = _gen(6)
    p = _params(g, 12, 6, 2, 2)
    x = _randn(g, 2, 9, 12)
    full = m.gqa_attention(x, p, 6, 2)
    cache = m.KVCache()
    parts = [m.gqa_decode_step(x[:, a:b], p, 6, 2, cache) for a, b in ((0, 3), (3, 7), (7, 8), (8, 9))]
    _close(torch.cat(parts, dim=1), full, atol=1e-10)
    assert len(cache) == 9 and cache.v.shape == (2, 2, 9, 2)


def test_matches_torch_sdpa_enable_gqa(m):
    g = _gen(7)
    p = _params(g, 16, 8, 2, 4)
    x = _randn(g, 3, 10, 16)
    _close(m.gqa_attention(x, p, 8, 2), _sdpa_reference(x, p, 8, 2), atol=1e-10)
    _close(m.gqa_attention(x, p, 8, 2, causal=False), _sdpa_reference(x, p, 8, 2, causal=False), atol=1e-10)
    # float32 输入应得到 float32 输出
    x32 = x.float()
    p32 = {name: w.float() for name, w in p.items()}
    _close(m.gqa_attention(x32, p32, 8, 2), _sdpa_reference(x32, p32, 8, 2), atol=1e-5)
