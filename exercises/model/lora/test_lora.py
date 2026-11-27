import torch
import torch.nn.functional as F

F64 = torch.float64


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _randn(g, *shape):
    return torch.randn(*shape, generator=g, dtype=F64)


def _close(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _make(m, out_f=6, in_f=5, r=2, alpha=8.0, seed=0):
    W = _randn(_gen(seed), out_f, in_f)
    return W, m.LoRALinear(W, r, alpha, _gen(seed + 100))


def test_init_matches_base(m):
    W, layer = _make(m)
    assert layer.A.shape == (2, 5) and layer.B.shape == (6, 2)
    assert layer.A.dtype == F64 and layer.B.dtype == F64, "A、B 的 dtype 应与 W 相同"
    assert bool((layer.B == 0).all()) and bool((layer.A != 0).any())
    assert layer.scaling == 4.0, f"scaling 应为 alpha / r = 4.0，实际 {layer.scaling}"
    assert layer.merged is False
    x = _randn(_gen(1), 3, 5)
    _close(layer.forward(x), x @ W.T, atol=1e-12)
    _close(layer.forward(x), F.linear(x, W), atol=1e-12)


def test_init_uses_generator(m):
    W = _randn(_gen(0), 4, 4000)
    a1 = m.LoRALinear(W, 4, 8.0, _gen(7)).A
    a2 = m.LoRALinear(W, 4, 8.0, _gen(7)).A
    a3 = m.LoRALinear(W, 4, 8.0, _gen(8)).A
    assert torch.equal(a1, a2), "同一个种子应得到同样的 A（随机数只能来自传入的 generator）"
    assert not torch.equal(a1, a3)
    # A ~ N(0, 标准差 1/r)：r=4 时标准差 0.25（不是方差 0.25 → 标准差 0.5）
    assert abs(a1.mean().item()) < 0.01, f"A 的均值应≈0，实际 {a1.mean().item():.4f}"
    assert abs(a1.std().item() - 0.25) < 0.01, f"A 的标准差应≈1/r=0.25，实际 {a1.std().item():.4f}"


def test_forward_formula(m):
    W, layer = _make(m)
    g = _gen(2)
    layer.A = _randn(g, 2, 5)
    layer.B = _randn(g, 6, 2)
    x = _randn(g, 4, 5)
    expected = x @ W.T + 4.0 * x @ layer.A.T @ layer.B.T
    _close(layer.forward(x), expected, atol=1e-12)
    _close(layer.forward(x), x @ (W + 4.0 * layer.B @ layer.A).T, atol=1e-12)  # 等价于合并后的权重
    x3 = _randn(g, 2, 3, 5)  # 任意前导维度
    _close(layer.forward(x3)[1, 2], layer.forward(x3[1, 2:3])[0], atol=1e-12)


def test_merge_equivalence_and_idempotent(m):
    W, layer = _make(m)
    g = _gen(3)
    layer.A = _randn(g, 2, 5)
    layer.B = _randn(g, 6, 2)
    x = _randn(g, 4, 5)
    before = layer.forward(x)
    layer.merge()
    assert layer.merged is True
    _close(layer.forward(x), before, atol=1e-10)
    _close(layer.W, W + 4.0 * layer.B @ layer.A, atol=1e-12)
    layer.merge()  # 再合并一次不应叠加
    _close(layer.forward(x), before, atol=1e-10)


def test_unmerge_restores(m):
    W, layer = _make(m)
    g = _gen(4)
    layer.unmerge()  # 未合并时什么也不做
    _close(layer.W, W, atol=0)
    layer.A = _randn(g, 2, 5)
    layer.B = _randn(g, 6, 2)
    layer.merge()
    layer.unmerge()
    assert layer.merged is False
    _close(layer.W, W, atol=1e-12)


def test_does_not_mutate_input(m):
    W, layer = _make(m)
    W_copy = W.clone()
    layer.B = torch.ones_like(layer.B)
    layer.merge()
    assert torch.equal(W, W_copy), "merge 改到了调用者传入的 W（构造时要复制）"


def test_param_count(m):
    _, layer = _make(m, out_f=512, in_f=256, r=4, alpha=8.0)
    n = layer.num_trainable_params()
    assert isinstance(n, int), "应返回 python int"
    assert n == 4 * (512 + 256)  # r·(out + in)，远小于 512·256


def test_only_lora_params_get_grads(m):
    g = _gen(5)
    W = _randn(g, 6, 5)
    W_param = W.clone().requires_grad_(True)  # 模拟来自 nn.Linear.weight 的、带梯度的基座权重
    layer = m.LoRALinear(W_param, 2, 8.0, _gen(105))
    assert layer.A.requires_grad and layer.B.requires_grad, "A、B 是可训练参数，requires_grad 应为 True"
    assert not layer.W.requires_grad, "基座权重 W 是冻结的，requires_grad 应为 False"
    x = _randn(g, 4, 5)
    target = _randn(g, 4, 6)

    def ref_grads(A, B):
        # 参照：对合并后的权重 W + s·B@A 用 autograd 求导
        A_ref = A.detach().clone().requires_grad_(True)
        B_ref = B.detach().clone().requires_grad_(True)
        ((x @ (W + 4.0 * B_ref @ A_ref).T - target) ** 2).sum().backward()
        return A_ref.grad, B_ref.grad

    # 刚初始化时 B=0：dL/dA ∝ B^T(...) = 0，只有 B 有非零梯度
    ((layer.forward(x) - target) ** 2).sum().backward()
    assert W_param.grad is None and layer.W.grad is None, "梯度不应流到基座权重 W"
    assert layer.A.grad is not None and layer.B.grad is not None
    gA, gB = ref_grads(layer.A, layer.B)
    _close(layer.A.grad, torch.zeros_like(layer.A), atol=0)
    assert layer.B.grad.abs().max().item() > 1e-3
    _close(layer.B.grad, gB, atol=1e-10)

    # A、B 都非零时，两者的梯度都应和参照一致
    layer.A = _randn(g, 2, 5).requires_grad_(True)
    layer.B = _randn(g, 6, 2).requires_grad_(True)
    ((layer.forward(x) - target) ** 2).sum().backward()
    gA, gB = ref_grads(layer.A, layer.B)
    _close(layer.A.grad, gA, atol=1e-10)
    _close(layer.B.grad, gB, atol=1e-10)
    assert W_param.grad is None and layer.W.grad is None


def test_merge_keeps_W_frozen(m):
    W, layer = _make(m)
    g = _gen(6)
    layer.A = _randn(g, 2, 5).requires_grad_(True)
    layer.B = _randn(g, 6, 2).requires_grad_(True)
    x = _randn(g, 3, 5)
    before = layer.forward(x).detach()
    layer.merge()
    assert not layer.W.requires_grad, "merge 要在 torch.no_grad() 下改 W，否则 W 会挂上 A、B 的计算图"
    _close(layer.forward(x), before, atol=1e-10)
    layer.unmerge()
    assert not layer.W.requires_grad
    _close(layer.W, W, atol=1e-12)
