import math

import torch

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _pos(x):
    return torch.tensor(x, dtype=torch.long)


def _randn(g, *shape, dtype=F64):
    return torch.randn(*shape, generator=g, dtype=dtype)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _rope_complex(x, positions, base=10000.0):
    """参照实现：第 i 维和第 i+d/2 维拼成复数 z_i，乘上 e^{i·p·θ_i}"""
    d = x.shape[-1]
    h = d // 2
    theta = _t([base ** (-2 * i / d) for i in range(h)])
    ang = positions.to(F64)[:, None] * theta  # (T, h)
    rot = torch.polar(torch.ones_like(ang), ang)
    z = torch.complex(x[..., :h].to(F64), x[..., h:].to(F64)) * rot
    return torch.cat([z.real, z.imag], dim=-1)


def test_cos_sin_table(m):
    cos, sin = m.rope_cos_sin(_pos([0, 1, 5]), dim=8, base=10000.0)
    assert cos.shape == (3, 8) and sin.shape == (3, 8)
    exp_cos = torch.zeros(3, 8, dtype=F64)
    exp_sin = torch.zeros(3, 8, dtype=F64)
    for pi, p in enumerate([0, 1, 5]):
        for j in range(8):
            theta = 10000.0 ** (-2 * (j % 4) / 8)
            exp_cos[pi, j] = math.cos(p * theta)
            exp_sin[pi, j] = math.sin(p * theta)
    _close(cos, exp_cos, atol=1e-12)  # 也检查 dtype：cos/sin 表应是 float64
    _close(sin, exp_sin, atol=1e-12)


def test_rotate_half(m):
    x = _t([1.0, 2.0, 3.0, 4.0])
    _close(m.rotate_half(x), _t([-3.0, -4.0, 1.0, 2.0]), atol=0)


def test_dim2_is_plain_rotation(m):
    # dim=2 时 θ0 = 1，(x0, x1) 绕原点转 p 弧度
    out = m.apply_rope(_t([[1.0, 0.0], [0.0, 2.0]]), _pos([1, 3]))
    _close(out[0], _t([math.cos(1), math.sin(1)]), atol=1e-12)
    _close(out[1], _t([-2 * math.sin(3), 2 * math.cos(3)]), atol=1e-12)


def test_half_split_pairing(m):
    # dim=4, base=10000：θ = [1, 0.01]；第 0 维和第 2 维配对，第 1 维和第 3 维配对
    e = torch.eye(4, dtype=F64)
    p = 2.0
    out = m.apply_rope(e, _pos([2, 2, 2, 2]))
    _close(out[0], _t([math.cos(p), 0, math.sin(p), 0]), atol=1e-12)
    _close(out[1], _t([0, math.cos(0.01 * p), 0, math.sin(0.01 * p)]), atol=1e-12)
    _close(out[2], _t([-math.sin(p), 0, math.cos(p), 0]), atol=1e-12)


def test_position_zero_is_identity_and_norm_preserved(m):
    g = _gen(0)
    x = _randn(g, 5, 16)
    _close(m.apply_rope(x[:1], _pos([0])), x[:1], atol=1e-14)
    y = m.apply_rope(x, torch.arange(5) * 37)
    _close(torch.linalg.vector_norm(y, dim=-1), torch.linalg.vector_norm(x, dim=-1), atol=1e-12)


def test_relative_position_property(m):
    g = _gen(1)
    q = _randn(g, 1, 32)
    k = _randn(g, 1, 32)

    def score(mq, nk):
        return (m.apply_rope(q, _pos([mq]))[0] @ m.apply_rope(k, _pos([nk]))[0]).item()

    base = score(5, 2)
    for s in (1, 10, 100, 1000):
        assert abs(score(5 + s, 2 + s) - base) < 1e-9, f"平移 {s} 后点积变了"
    assert abs(score(5, 2) - score(5, 3)) > 1e-6


def test_broadcast_over_batch_and_heads(m):
    g = _gen(2)
    x = _randn(g, 2, 3, 4, 8)  # (B, H, T, dim)
    pos = _pos([3, 4, 5, 6])
    y = m.apply_rope(x, pos)
    assert y.shape == x.shape
    for b in range(2):
        for h in range(3):
            _close(y[b, h], m.apply_rope(x[b, h], pos), atol=1e-14)


def test_matches_complex_formula(m):
    g = _gen(3)
    x = _randn(g, 2, 3, 7, 16)
    pos = _pos([0, 3, 10, 100, 1000, 4096, 9])
    for base in (10000.0, 500000.0):
        _close(m.apply_rope(x, pos, base=base), _rope_complex(x, pos, base), atol=1e-10)
    # float32 输入：输出保持 float32
    x32 = x.float()
    pos_small = _pos([0, 1, 2, 5, 8, 13, 21])
    _close(m.apply_rope(x32, pos_small), _rope_complex(x32, pos_small).float(), atol=1e-5)
