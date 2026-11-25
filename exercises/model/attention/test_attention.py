import math

import numpy as np


def _loop_reference(q, k, v, causal, kpm):
    """逐个 query 用纯 Python 循环按定义计算。"""
    B, H, Tq, D = q.shape
    Tk = k.shape[2]
    out = np.zeros((B, H, Tq, v.shape[-1]))
    att = np.zeros((B, H, Tq, Tk))
    for b in range(B):
        for h in range(H):
            for i in range(Tq):
                s = []
                for j in range(Tk):
                    ok = not (causal and j > i + Tk - Tq)
                    if kpm is not None and not kpm[b, j]:
                        ok = False
                    s.append(float(np.dot(q[b, h, i], k[b, h, j])) / math.sqrt(D) if ok else None)
                mx = max(x for x in s if x is not None)
                w = [math.exp(x - mx) if x is not None else 0.0 for x in s]
                z = math.fsum(w)
                w = [x / z for x in w]
                att[b, h, i] = w
                out[b, h, i] = sum(w[j] * v[b, h, j] for j in range(Tk))
    return out, att


def _rand(rng, *shape):
    return rng.normal(size=shape)


def test_matches_loop_reference_plain(m):
    rng = np.random.default_rng(0)
    q, k, v = _rand(rng, 2, 3, 4, 8), _rand(rng, 2, 3, 6, 8), _rand(rng, 2, 3, 6, 5)
    out, att = m.scaled_dot_product_attention(q, k, v)
    ro, ra = _loop_reference(q, k, v, False, None)
    np.testing.assert_allclose(att, ra, atol=1e-12)
    np.testing.assert_allclose(out, ro, atol=1e-12)


def test_causal_matches_reference_and_zeros(m):
    rng = np.random.default_rng(1)
    q, k, v = _rand(rng, 1, 2, 5, 4), _rand(rng, 1, 2, 5, 4), _rand(rng, 1, 2, 5, 4)
    out, att = m.scaled_dot_product_attention(q, k, v, causal=True)
    ro, ra = _loop_reference(q, k, v, True, None)
    np.testing.assert_allclose(out, ro, atol=1e-12)
    upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
    assert np.all(att[..., upper] == 0.0), "causal 时上三角权重必须严格为 0"
    np.testing.assert_allclose(att[..., 0, 0], 1.0)  # 第一个 query 只能看自己


def test_rows_sum_to_one(m):
    rng = np.random.default_rng(2)
    q, k, v = _rand(rng, 2, 2, 7, 3), _rand(rng, 2, 2, 7, 3), _rand(rng, 2, 2, 7, 3)
    kpm = np.array([[True] * 7, [True] * 4 + [False] * 3])
    _, att = m.scaled_dot_product_attention(q, k, v, causal=True, key_padding_mask=kpm)
    np.testing.assert_allclose(att.sum(axis=-1), 1.0, atol=1e-12)


def test_padding_keys_are_ignored(m):
    rng = np.random.default_rng(3)
    q, k, v = _rand(rng, 2, 2, 3, 4), _rand(rng, 2, 2, 6, 4), _rand(rng, 2, 2, 6, 4)
    kpm = np.array([[True, True, True, True, False, False], [False, True, True, True, True, True]])
    out, att = m.scaled_dot_product_attention(q, k, v, key_padding_mask=kpm)
    # 把 padding 位置的 k/v 改成垃圾，输出不应变化
    k2, v2 = k.copy(), v.copy()
    k2[0, :, 4:] = 1e3
    v2[0, :, 4:] = -1e3
    k2[1, :, 0] = -1e3
    v2[1, :, 0] = 1e3
    out2, att2 = m.scaled_dot_product_attention(q, k2, v2, key_padding_mask=kpm)
    np.testing.assert_allclose(out, out2, atol=1e-12)
    assert np.all(att[0, :, :, 4:] == 0.0) and np.all(att[1, :, :, 0] == 0.0)
    ro, _ = _loop_reference(q, k, v, False, kpm)
    np.testing.assert_allclose(out, ro, atol=1e-12)


def test_scale_hand_computed(m):
    # d=4：q·k0 = 4 → 4/2=2，q·k1 = 0 → 0；权重 = softmax([2, 0])
    q = np.array([[[[1.0, 1.0, 1.0, 1.0]]]])
    k = np.array([[[[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]]]])
    v = np.array([[[[10.0], [0.0]]]])
    out, att = m.scaled_dot_product_attention(q, k, v)
    w0 = math.exp(2) / (math.exp(2) + 1.0)
    np.testing.assert_allclose(att[0, 0, 0], [w0, 1 - w0], atol=1e-12)
    np.testing.assert_allclose(out[0, 0, 0, 0], 10 * w0, atol=1e-12)


def test_single_query_decode_sees_all_keys(m):
    # Tq=1 的 causal（右下角对齐）应当等价于不加 mask
    rng = np.random.default_rng(4)
    q, k, v = _rand(rng, 1, 2, 1, 4), _rand(rng, 1, 2, 5, 4), _rand(rng, 1, 2, 5, 4)
    a, _ = m.scaled_dot_product_attention(q, k, v, causal=True)
    b, _ = m.scaled_dot_product_attention(q, k, v, causal=False)
    np.testing.assert_allclose(a, b, atol=1e-12)
    # Tq=2, Tk=5：第一个 query 能看前 4 个 key
    q2 = _rand(rng, 1, 1, 2, 4)
    _, att = m.scaled_dot_product_attention(q2, k[:, :1], v[:, :1], causal=True)
    assert att[0, 0, 0, 4] == 0.0 and att[0, 0, 0, 3] > 0.0 and att[0, 0, 1, 4] > 0.0
