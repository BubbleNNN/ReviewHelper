import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _loop_reference(q, k, v, causal, kpm):
    """逐个 query 用纯 Python 循环（python float）按定义计算。kpm: (B, Tk) bool tensor 或 None。"""
    B, H, Tq, D = q.shape
    Tk, Dv = k.shape[2], v.shape[-1]
    Q, K, Vv = q.tolist(), k.tolist(), v.tolist()
    M = kpm.tolist() if kpm is not None else None
    out = torch.zeros(B, H, Tq, Dv, dtype=F64)
    att = torch.zeros(B, H, Tq, Tk, dtype=F64)
    for b in range(B):
        for h in range(H):
            for i in range(Tq):
                s = []
                for j in range(Tk):
                    ok = not (causal and j > i + Tk - Tq)
                    if M is not None and not M[b][j]:
                        ok = False
                    dot = math.fsum(x * y for x, y in zip(Q[b][h][i], K[b][h][j]))
                    s.append(dot / math.sqrt(D) if ok else None)
                mx = max(x for x in s if x is not None)
                w = [math.exp(x - mx) if x is not None else 0.0 for x in s]
                z = math.fsum(w)
                w = [x / z for x in w]
                att[b, h, i] = _t(w)
                out[b, h, i] = _t([math.fsum(w[j] * Vv[b][h][j][c] for j in range(Tk)) for c in range(Dv)])
    return out, att


def _rand(g, *shape):
    return torch.randn(*shape, generator=g, dtype=F64)


def test_matches_loop_reference_plain(m):
    g = _gen(0)
    q, k, v = _rand(g, 2, 3, 4, 8), _rand(g, 2, 3, 6, 8), _rand(g, 2, 3, 6, 5)
    out, att = m.scaled_dot_product_attention(q, k, v)
    ro, ra = _loop_reference(q, k, v, False, None)
    _close(att, ra, atol=1e-12)
    _close(out, ro, atol=1e-12)


def test_causal_matches_reference_and_zeros(m):
    g = _gen(1)
    q, k, v = _rand(g, 1, 2, 5, 4), _rand(g, 1, 2, 5, 4), _rand(g, 1, 2, 5, 4)
    out, att = m.scaled_dot_product_attention(q, k, v, causal=True)
    ro, ra = _loop_reference(q, k, v, True, None)
    _close(out, ro, atol=1e-12)
    _close(att, ra, atol=1e-12)
    upper = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
    assert (att[..., upper] == 0.0).all(), "causal 时上三角权重必须严格为 0"
    _close(att[..., 0, 0], torch.ones(1, 2, dtype=F64), atol=1e-15)  # 第一个 query 只能看自己


def test_rows_sum_to_one(m):
    g = _gen(2)
    q, k, v = _rand(g, 2, 2, 7, 3), _rand(g, 2, 2, 7, 3), _rand(g, 2, 2, 7, 3)
    kpm = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    _, att = m.scaled_dot_product_attention(q, k, v, causal=True, key_padding_mask=kpm)
    _close(att.sum(dim=-1), torch.ones(2, 2, 7, dtype=F64), atol=1e-12)


def test_padding_keys_are_ignored(m):
    g = _gen(3)
    q, k, v = _rand(g, 2, 2, 3, 4), _rand(g, 2, 2, 6, 4), _rand(g, 2, 2, 6, 4)
    kpm = torch.tensor([[True, True, True, True, False, False], [False, True, True, True, True, True]])
    out, att = m.scaled_dot_product_attention(q, k, v, key_padding_mask=kpm)
    # 把 padding 位置的 k/v 改成垃圾，输出不应变化
    k2, v2 = k.clone(), v.clone()
    k2[0, :, 4:] = 1e3
    v2[0, :, 4:] = -1e3
    k2[1, :, 0] = -1e3
    v2[1, :, 0] = 1e3
    out2, att2 = m.scaled_dot_product_attention(q, k2, v2, key_padding_mask=kpm)
    _close(out, out2, atol=1e-12)
    assert (att[0, :, :, 4:] == 0.0).all() and (att[1, :, :, 0] == 0.0).all()
    ro, _ = _loop_reference(q, k, v, False, kpm)
    _close(out, ro, atol=1e-12)


def test_scale_hand_computed(m):
    # d=4：q·k0 = 4 → 4/2=2，q·k1 = 0 → 0；权重 = softmax([2, 0])
    q = _t([[[[1.0, 1.0, 1.0, 1.0]]]])
    k = _t([[[[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]]]])
    v = _t([[[[10.0], [0.0]]]])
    out, att = m.scaled_dot_product_attention(q, k, v)
    w0 = math.exp(2) / (math.exp(2) + 1.0)
    _close(att[0, 0, 0], _t([w0, 1 - w0]), atol=1e-12)
    _close(out[0, 0, 0, 0], _t(10 * w0), atol=1e-12)


def test_single_query_decode_sees_all_keys(m):
    # Tq=1 的 causal（右下角对齐）应当等价于不加 mask
    g = _gen(4)
    q, k, v = _rand(g, 1, 2, 1, 4), _rand(g, 1, 2, 5, 4), _rand(g, 1, 2, 5, 4)
    a, _ = m.scaled_dot_product_attention(q, k, v, causal=True)
    b, _ = m.scaled_dot_product_attention(q, k, v, causal=False)
    _close(a, b, atol=1e-12)
    # Tq=2, Tk=5：第一个 query 能看前 4 个 key
    q2 = _rand(g, 1, 1, 2, 4)
    _, att = m.scaled_dot_product_attention(q2, k[:, :1], v[:, :1], causal=True)
    assert att[0, 0, 0, 4].item() == 0.0 and att[0, 0, 0, 3].item() > 0.0 and att[0, 0, 1, 4].item() > 0.0


def test_matches_torch_sdpa(m):
    # torch 自带的 F.scaled_dot_product_attention 当对照（题目里不许直接调用）。它的 attn_mask 也是 True=可以看
    g = _gen(5)
    B, H, Tq, Tk, D, Dv = 2, 3, 4, 7, 8, 5
    q, k, v = _rand(g, B, H, Tq, D), _rand(g, B, H, Tk, D), _rand(g, B, H, Tk, Dv)
    kpm = torch.ones(B, Tk, dtype=torch.bool)
    kpm[0, :2] = False  # 左 padding
    kpm[1, -1] = False  # 右 padding
    out, _ = m.scaled_dot_product_attention(q, k, v)
    _close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-12)
    # Tq == Tk 时右下角对齐与左上角对齐一样，可以直接用 is_causal
    qs = _rand(g, B, H, Tk, D)
    out, _ = m.scaled_dot_product_attention(qs, k, v, causal=True)
    _close(out, F.scaled_dot_product_attention(qs, k, v, is_causal=True), atol=1e-12)
    # Tq != Tk：is_causal 是左上角对齐，要自己给右下角对齐的 mask，再叠加 padding
    bottom_right = torch.ones(Tq, Tk, dtype=torch.bool).tril(diagonal=Tk - Tq)
    out, _ = m.scaled_dot_product_attention(q, k, v, causal=True, key_padding_mask=kpm)
    _close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=bottom_right & kpm[:, None, None, :]), atol=1e-12)
    # float32 输入应得到 float32 输出
    q32, k32, v32 = q.float(), k.float(), v.float()
    out32, att32 = m.scaled_dot_product_attention(q32, k32, v32, key_padding_mask=kpm)
    _close(out32, F.scaled_dot_product_attention(q32, k32, v32, attn_mask=kpm[:, None, None, :]), atol=1e-5)
    assert att32.dtype == torch.float32
