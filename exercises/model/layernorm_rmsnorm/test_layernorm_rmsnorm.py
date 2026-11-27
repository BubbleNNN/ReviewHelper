import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _randn(g, *shape, dtype=F64):
    return torch.randn(*shape, generator=g, dtype=dtype)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def test_layernorm_hand_computed(m):
    # x=[1,2,3]：μ=2，σ²=2/3，x̂ = [-1, 0, 1] / sqrt(2/3)
    y, _ = m.layernorm_forward(_t([[1.0, 2.0, 3.0]]), torch.ones(3, dtype=F64), torch.zeros(3, dtype=F64), eps=0.0)
    s = 1.0 / math.sqrt(2.0 / 3.0)
    _close(y, _t([[-s, 0.0, s]]), atol=1e-12)


def test_layernorm_normalizes_and_affine(m):
    g = _gen(0)
    x = 3.0 + 5.0 * _randn(g, 6, 16)
    y, _ = m.layernorm_forward(x, torch.ones(16, dtype=F64), torch.zeros(16, dtype=F64), eps=1e-5)
    _close(y.mean(dim=1), torch.zeros(6, dtype=F64), atol=1e-10)
    _close((y ** 2).mean(dim=1), torch.ones(6, dtype=F64), atol=1e-5)  # 有偏方差≈1
    gamma = _randn(g, 16)
    beta = _randn(g, 16)
    y2, _ = m.layernorm_forward(x, gamma, beta, eps=1e-5)
    _close(y2, y * gamma + beta, atol=1e-10)


def test_layernorm_matches_torch(m):
    g = _gen(5)
    x = 2.0 + 3.0 * _randn(g, 5, 12)
    gamma = _randn(g, 12)
    beta = _randn(g, 12)
    for eps in (1e-5, 0.1):
        y, _ = m.layernorm_forward(x, gamma, beta, eps=eps)
        _close(y, F.layer_norm(x, (12,), gamma, beta, eps=eps), atol=1e-12)
    # float32 输入，输出也应是 float32
    x32, g32, b32 = x.float(), gamma.float(), beta.float()
    y32, _ = m.layernorm_forward(x32, g32, b32)
    _close(y32, F.layer_norm(x32, (12,), g32, b32, eps=1e-5), atol=1e-5)


def test_rmsnorm_hand_computed(m):
    # x=[3,4]：mean(x²)=12.5，rms=sqrt(12.5)
    r = math.sqrt(12.5)
    y = m.rmsnorm(_t([[3.0, 4.0]]), _t([1.0, 1.0]), eps=0.0)
    _close(y, _t([[3 / r, 4 / r]]), atol=1e-12)
    y2 = m.rmsnorm(_t([[3.0, 4.0]]), _t([2.0, 0.5]), eps=0.0)
    _close(y2, _t([[6 / r, 2 / r]]), atol=1e-12)


def test_rmsnorm_properties(m):
    g = _gen(1)
    x = _randn(g, 2, 3, 8)
    w = torch.ones(8, dtype=F64)
    y = m.rmsnorm(x, w, eps=0.0)
    assert y.shape == x.shape
    _close(m.rmsnorm(7.5 * x, w, eps=0.0), y, atol=1e-12)  # 正缩放不变
    _close((y ** 2).mean(dim=-1), torch.ones(2, 3, dtype=F64), atol=1e-12)
    # 不减均值：整体平移会改变输出（和 LayerNorm 的区别）
    assert not torch.allclose(m.rmsnorm(x + 3.0, w, eps=0.0), y)


def test_rmsnorm_matches_torch(m):
    g = _gen(6)
    x = _randn(g, 2, 4, 10)
    w = _randn(g, 10)
    for eps in (1e-6, 0.5):
        _close(m.rmsnorm(x, w, eps=eps), F.rms_norm(x, (10,), w, eps=eps), atol=1e-12)
    x32, w32 = x.float(), w.float()
    _close(m.rmsnorm(x32, w32), F.rms_norm(x32, (10,), w32, eps=1e-6), atol=1e-5)


def _numeric_grads(m, x, gamma, beta, G, eps, h=1e-6):
    def loss(x_, g_, b_):
        y, _ = m.layernorm_forward(x_, g_, b_, eps=eps)
        return float((y * G).sum())

    grads = []
    for arr_idx, arr in enumerate((x, gamma, beta)):
        g = torch.zeros_like(arr)
        for i in range(arr.numel()):
            args_p = [x.clone(), gamma.clone(), beta.clone()]
            args_m = [x.clone(), gamma.clone(), beta.clone()]
            args_p[arr_idx].view(-1)[i] += h
            args_m[arr_idx].view(-1)[i] -= h
            g.view(-1)[i] = (loss(*args_p) - loss(*args_m)) / (2 * h)
        grads.append(g)
    return grads


def test_layernorm_backward_finite_difference(m):
    g = _gen(2)
    x = _randn(g, 3, 5)
    gamma = _randn(g, 5)
    beta = _randn(g, 5)
    G = _randn(g, 3, 5)  # loss = Σ y ⊙ G，所以 dy = G
    _, cache = m.layernorm_forward(x, gamma, beta, eps=1e-5)
    dx, dgamma, dbeta = m.layernorm_backward(G, cache)
    ndx, ndgamma, ndbeta = _numeric_grads(m, x, gamma, beta, G, eps=1e-5)
    _close(dx, ndx, atol=1e-6)
    _close(dgamma, ndgamma, atol=1e-6)
    _close(dbeta, ndbeta, atol=1e-6)


def test_layernorm_backward_matches_autograd(m):
    # 参照：对 F.layer_norm 做 autograd，loss = Σ y ⊙ G
    g = _gen(7)
    N, D = 6, 9
    x = 1.5 + 2.0 * _randn(g, N, D)
    gamma = _randn(g, D)
    beta = _randn(g, D)
    G = _randn(g, N, D)
    xr, gr, br = (t.clone().requires_grad_(True) for t in (x, gamma, beta))
    (F.layer_norm(xr, (D,), gr, br, eps=1e-5) * G).sum().backward()

    # 手写的前向/反向在 no_grad 下调用：不能依赖 autograd
    with torch.no_grad():
        _, cache = m.layernorm_forward(x, gamma, beta, eps=1e-5)
        dx, dgamma, dbeta = m.layernorm_backward(G, cache)
    _close(dx, xr.grad, atol=1e-10)
    _close(dgamma, gr.grad, atol=1e-10)
    _close(dbeta, br.grad, atol=1e-10)


def test_layernorm_backward_shift_invariance(m):
    # LN(x + c·1) = LN(x)，所以 dx 沿 D 维求和恒为 0
    g = _gen(3)
    x = _randn(g, 4, 7)
    _, cache = m.layernorm_forward(x, _randn(g, 7), _randn(g, 7))
    dx, _, _ = m.layernorm_backward(_randn(g, 4, 7), cache)
    _close(dx.sum(dim=1), torch.zeros(4, dtype=F64), atol=1e-10)
