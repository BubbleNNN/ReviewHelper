import math

import torch

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _long(x):
    return torch.tensor(x, dtype=torch.long)


def _close(actual, expected, atol=1e-12):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def _eq(actual, expected):
    """整数 / bool tensor 精确相等（同时检查 dtype）"""
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _distinct_idx(g, T, E, k):
    """每个 token 选 k 个互不相同的专家（随机打分取前 k）"""
    return torch.rand(T, E, generator=g).argsort(dim=-1)[:, :k]


def test_router_probs(m):
    p = m.router_probs(_t([[0.0, 0.0], [1000.0, 0.0]]))
    _close(p, _t([[0.5, 0.5], [1.0, 0.0]]))


def test_top_k_hand(m):
    idx, w = m.top_k_routing(_t([[1.0, 2.0, 3.0, 0.0], [0.5, 0.5, 0.0, 0.5]]), 2)
    _eq(idx[0], _long([2, 1]))
    s = 1 / (1 + math.exp(-1))  # e³/(e³+e²) = σ(1)
    _close(w[0], _t([s, 1 - s]))
    _eq(idx[1], _long([0, 1]))  # 三个并列，取下标小的
    _close(w[1], _t([0.5, 0.5]))
    idx1, w1 = m.top_k_routing(_t([[0.1, 0.9, 0.3]]), 1)
    _eq(idx1, _long([[1]]))
    _close(w1, _t([[1.0]]))


def test_capacity(m):
    assert m.expert_capacity(10, 4, 2, 1.0) == 5
    assert m.expert_capacity(10, 4, 2, 1.25) == 7   # ceil(6.25)
    assert m.expert_capacity(16, 8, 1, 1.0) == 2


def test_dispatch_slot_priority(m):
    idx = _long([[0, 1], [0, 2], [0, 1], [1, 0]])
    assign, dropped = m.dispatch(idx, 3, capacity=2)
    # 第 1 轮（slot 0）：t0→e0, t1→e0, t2→e0 满 → 丢, t3→e1
    # 第 2 轮（slot 1）：t0→e1, t1→e2, t2→e1 满 → 丢, t3→e0 满 → 丢
    assert [list(a) for a in assign] == [[0, 1], [3, 0], [1]]
    _eq(dropped, torch.tensor([[False, False], [False, False], [True, True], [False, True]]))


def test_aux_loss_uniform_and_collapsed(m):
    logits = torch.zeros(8, 4, dtype=F64)
    idx = _long([[0], [1], [2], [3], [0], [1], [2], [3]])
    assert abs(m.load_balancing_loss(logits, idx, 4) - 1.0) < 1e-12
    logits2 = torch.log(_t([0.9, 0.1])).repeat(4, 1)
    idx2 = torch.zeros(4, 1, dtype=torch.long)
    assert abs(m.load_balancing_loss(logits2, idx2, 2) - 1.8) < 1e-12  # 2·(1·0.9 + 0·0.1)


def test_aux_loss_counts_all_k_slots(m):
    logits = torch.zeros(2, 4, dtype=F64)  # P = 1/4
    idx = _long([[0, 1], [0, 2]])  # f = [2/4, 1/4, 1/4, 0]
    assert abs(m.load_balancing_loss(logits, idx, 4) - 4 * 0.25 * 1.0) < 1e-12


def test_moe_forward_combine_and_drop(m):
    g = torch.Generator().manual_seed(0)
    T, d, E = 6, 3, 3
    x = torch.randn(T, d, generator=g, dtype=F64)
    logits = torch.randn(T, E, generator=g, dtype=F64)
    experts = [lambda z, c=c: z * (c + 1.0) for c in range(E)]  # 专家 e 把输入乘 e+1
    idx, w = m.top_k_routing(logits, 2)
    y = m.moe_forward(x, logits, experts, k=2, capacity=100)
    expected = torch.stack([(w[t, 0] * (idx[t, 0] + 1) + w[t, 1] * (idx[t, 1] + 1)) * x[t] for t in range(T)])
    _close(y, expected)
    # 容量 1：每个专家只收 1 个 token，其余 slot 贡献 0
    y1 = m.moe_forward(x, logits, experts, k=2, capacity=1)
    _, dropped = m.dispatch(idx, E, 1)
    exp1 = torch.stack([sum(0.0 if dropped[t, j] else w[t, j] * (idx[t, j] + 1) for j in range(2)) * x[t]
                        for t in range(T)])
    _close(y1, exp1)
    assert int(dropped.sum()) == 2 * T - len(set(idx.flatten().tolist()))  # 每个被选中的专家恰好收 1 个


def test_top_k_matches_torch_topk(m):
    # 随机 logits 几乎不会并列，这时 torch.topk 的结果可以直接当参照
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(16, 8, generator=g, dtype=F64) * 2
    probs = torch.softmax(logits, dim=-1)
    for k in (1, 2, 3, 8):
        idx, w = m.top_k_routing(logits, k)
        ref = torch.topk(probs, k, dim=-1, sorted=True)
        _eq(idx, ref.indices)
        _close(w, ref.values / ref.values.sum(dim=-1, keepdim=True))


def test_dispatch_matches_brute_force(m):
    # 按定义判丢弃：(t, j) 被丢 ⇔ 优先级在它之前（slot 更小，或同 slot 下 token 更靠前）且选了同一专家的 slot 数 ≥ capacity
    g = torch.Generator().manual_seed(2)
    T, E, k = 12, 4, 2
    for trial in range(3):
        idx = _distinct_idx(g, T, E, k)
        e_of = idx.tolist()
        for cap in (0, 1, 3, 5, 6, T * k):
            assign, dropped = m.dispatch(idx, E, cap)
            assert dropped.shape == (T, k) and dropped.dtype == torch.bool
            exp_drop = torch.zeros(T, k, dtype=torch.bool)
            for t in range(T):
                for j in range(k):
                    ahead = sum(1 for t2 in range(T) for j2 in range(k)
                                if e_of[t2][j2] == e_of[t][j] and (j2, t2) < (j, t))
                    exp_drop[t, j] = ahead >= cap
            _eq(dropped, exp_drop)
            for e in range(E):
                kept = sorted((j, t) for t in range(T) for j in range(k) if e_of[t][j] == e and not exp_drop[t, j])
                assert list(assign[e]) == [t for _, t in kept], f"trial {trial}, cap {cap}, 专家 {e} 的 token 顺序不对"
                assert len(assign[e]) <= cap
            assert sum(len(a) for a in assign) + int(dropped.sum()) == T * k


def test_aux_loss_matches_formula(m):
    # 测试方按定义直接算：f_i 数个数，P_i = exp(z)/Σexp(z) 在 token 上求均值
    g = torch.Generator().manual_seed(3)
    T, E, k = 10, 5, 2
    logits = torch.randn(T, E, generator=g, dtype=F64)
    idx = _distinct_idx(g, T, E, k)
    flat = idx.flatten().tolist()
    f = [flat.count(i) / (T * k) for i in range(E)]
    ez = torch.exp(logits)
    P = (ez / ez.sum(dim=-1, keepdim=True)).mean(dim=0).tolist()
    expected = E * sum(fi * pi for fi, pi in zip(f, P))
    loss = m.load_balancing_loss(logits, idx, E)
    assert torch.is_tensor(loss) and loss.dim() == 0, "load_balancing_loss 应返回 0-d tensor（要加进训练 loss）"
    assert abs(loss.item() - expected) < 1e-12


def test_aux_loss_gradient(m):
    # L = (E/T) Σ_t Σ_i f_i p_ti，f 是常数；∂L/∂z_tj = (E/T) · p_tj · (f_j - Σ_i f_i p_ti)
    g = torch.Generator().manual_seed(5)
    T, E, k = 6, 4, 2
    logits = torch.randn(T, E, generator=g, dtype=F64).requires_grad_(True)
    idx = _distinct_idx(g, T, E, k)
    loss = m.load_balancing_loss(logits, idx, E)
    loss.backward()
    assert logits.grad is not None, "aux loss 要能把梯度传回 router logits"
    flat = idx.flatten().tolist()
    f = _t([flat.count(i) / (T * k) for i in range(E)])
    with torch.no_grad():
        ez = torch.exp(logits)
        p = ez / ez.sum(dim=-1, keepdim=True)
        expected = (E / T) * p * (f[None, :] - (p * f[None, :]).sum(dim=-1, keepdim=True))
    _close(logits.grad, expected)


def test_float32_dtype_preserved(m):
    g = torch.Generator().manual_seed(4)
    x = torch.randn(5, 3, generator=g)  # float32
    logits = torch.randn(5, 4, generator=g)
    assert m.router_probs(logits).dtype == torch.float32
    idx, w = m.top_k_routing(logits, 2)
    assert idx.dtype == torch.long, "expert_idx 应是 long tensor"
    assert w.dtype == torch.float32, "weights 的 dtype 应与 logits 一致"
    y = m.moe_forward(x, logits, [lambda z: 2 * z] * 4, k=2, capacity=100)
    _close(y, 2 * x, atol=1e-5)  # 每个专家都乘 2，权重和为 1
