import math

import torch

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _close(actual, expected, atol=1e-12):
    torch.testing.assert_close(actual, expected if torch.is_tensor(expected) else _t(expected), atol=atol, rtol=0)


def _ones_mask(*shape):
    return torch.ones(*shape, dtype=torch.bool)


def _one(m, A, r, **kw):
    """单个 token 的损失：old=0，new=log r。"""
    loss, met = m.ppo_policy_loss(torch.zeros(1, 1, dtype=F64), _t([[math.log(r)]]), _t([[A]]),
                                  _ones_mask(1, 1), **kw)
    return loss, met


def test_clip_regions_positive_advantage(m):
    loss, met = _one(m, 1.0, 1.5)   # r 超过 1.2：被截到 -1.2
    _close(loss, -1.2)
    assert met["pg_clipfrac"] == 1.0
    loss, met = _one(m, 1.0, 0.5)   # r<1 的好动作：不截断
    _close(loss, -0.5)
    assert met["pg_clipfrac"] == 0.0
    loss, _ = _one(m, 2.0, 5.0)
    _close(loss, -2.4)


def test_clip_regions_negative_advantage_and_dual_clip(m):
    loss, met = _one(m, -1.0, 0.5)  # 坏动作概率已经降够：截到 0.8
    _close(loss, 0.8)
    assert met["pg_clipfrac"] == 1.0 and met["pg_clipfrac_lower"] == 0.0
    loss, met = _one(m, -1.0, 1.5)  # r 在 (1, c) 之间：正常梯度
    _close(loss, 1.5)
    assert met["pg_clipfrac"] == 0.0 and met["pg_clipfrac_lower"] == 0.0
    loss, met = _one(m, -1.0, 5.0)  # r 超过 c=3：dual-clip 截到 3
    _close(loss, 3.0)
    assert met["pg_clipfrac_lower"] == 1.0
    loss, _ = _one(m, -1.0, 5.0, clip_ratio_c=10.0)
    _close(loss, 5.0)


def test_clip_higher(m):
    loss, _ = _one(m, 1.0, 1.25, clip_ratio_low=0.2, clip_ratio_high=0.28)
    _close(loss, -1.25)
    loss, _ = _one(m, 1.0, 1.4, clip_ratio_low=0.2, clip_ratio_high=0.28)
    _close(loss, -1.28)
    loss, _ = _one(m, -1.0, 0.75, clip_ratio_low=0.2, clip_ratio_high=0.28)
    _close(loss, 0.8)


def test_agg_modes_by_hand(m):
    L = _t([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0]])
    M = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)  # 第 3 条整行无效
    _close(m.agg_loss(L, M, "token-mean"), 10 / 4)
    _close(m.agg_loss(L, M, "seq-mean-token-sum"), (6 + 4) / 2)
    _close(m.agg_loss(L, M, "seq-mean-token-mean"), (2 + 4) / 2, atol=1e-6)
    _close(m.agg_loss(L, M, "seq-mean-token-sum-norm"), 5 / 4)
    _close(m.agg_loss(L, M, "seq-mean-token-sum-norm", loss_scale_factor=10), 0.5)


def test_invalid_mode_raises(m):
    try:
        m.agg_loss(torch.ones(1, 2, dtype=F64), _ones_mask(1, 2), "bogus-mode")
    except ValueError:
        return
    raise AssertionError("非法 loss_agg_mode 应抛 ValueError")


def test_batch_metrics_and_mask(m):
    # 第 0 行：四种情况各一个 token；第 1 行：只有第 0 个 token 有效，其余是垃圾
    r = _t([[1.5, 0.5, 0.5, 5.0], [1.0, 50.0, 1e-9, 7.0]])
    A = _t([[1.0, 1.0, -1.0, -1.0], [2.0, -9.0, 9.0, -9.0]])
    M = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    loss, met = m.ppo_policy_loss(torch.zeros(2, 4, dtype=F64), torch.log(r), A, M)
    # 有效 token 损失：-1.2, -0.5, 0.8, 3.0, -2.0
    _close(loss, (-1.2 - 0.5 + 0.8 + 3.0 - 2.0) / 5)
    # L2 > L1 的只有 (A=1, r=1.5) 和 (A=-1, r=0.5)；(A=-1, r=5) 是被 dual-clip 截的，算在 lower 里
    assert abs(met["pg_clipfrac"] - 2 / 5) < 1e-12
    assert abs(met["pg_clipfrac_lower"] - 1 / 5) < 1e-12
    assert abs(met["ppo_kl"] - (-math.log(1.5) - 2 * math.log(0.5) - math.log(5.0)) / 5) < 1e-12
    loss2, _ = m.ppo_policy_loss(torch.zeros(2, 4, dtype=F64), torch.log(r), A, M, loss_agg_mode="seq-mean-token-mean")
    _close(loss2, ((-1.2 - 0.5 + 0.8 + 3.0) / 4 + (-2.0)) / 2, atol=1e-6)


def test_log_ratio_is_clamped(m):
    old = _t([[0.0, 0.0]])
    new = _t([[50.0, -50.0]])
    loss, met = m.ppo_policy_loss(old, new, _t([[1.0, -1.0]]), _ones_mask(1, 2))
    assert torch.isfinite(loss)
    assert abs(met["ppo_kl"] - 0.0) < 1e-12  # (-20 + 20) / 2
    loss1, met1 = m.ppo_policy_loss(old[:, :1], new[:, :1], _t([[1.0]]), _ones_mask(1, 1))
    assert abs(met1["ppo_kl"] - (-20.0)) < 1e-12
    _close(loss1, -1.2)


def test_backward_only_through_unclipped_valid_tokens(m):
    # 每列一种情况（old=0，所以 r = exp(log_prob)）：
    #   0: A=+1, r=1.5  上截断      → 梯度 0
    #   1: A=+1, r=0.5  正常        → -A·r/N
    #   2: A=-1, r=0.5  下截断      → 0
    #   3: A=-1, r=1.5  正常        → -A·r/N
    #   4: A=-1, r=5    dual-clip   → 0
    #   5: A=+2, r=1.1  在 clip 区间内 → -A·r/N
    #   6, 7: mask=False（本来属于“正常”情况）→ 0
    r = _t([[1.5, 0.5, 0.5, 1.5, 5.0, 1.1, 0.9, 2.0]])
    A = _t([[1.0, 1.0, -1.0, -1.0, -1.0, 2.0, 1.0, -3.0]])
    M = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    N = 6
    old = torch.zeros(1, 8, dtype=F64, requires_grad=True)
    adv = A.clone().requires_grad_(True)
    log_prob = torch.log(r).requires_grad_(True)
    loss, met = m.ppo_policy_loss(old, log_prob, adv, M)
    assert loss.dim() == 0 and loss.dtype == F64, "loss 应是 float64 的 0-d tensor"
    assert all(isinstance(v, float) for v in met.values()), "metrics 应是 python float"
    loss.backward()
    unclipped = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 0]], dtype=torch.bool)
    expected = torch.where(unclipped, -A * r / N, 0.0)
    _close(log_prob.grad, expected)
    assert old.grad is None or torch.all(old.grad == 0), "old_log_prob 是常数，不该有梯度（detach）"
    assert adv.grad is None or torch.all(adv.grad == 0), "advantages 是常数，不该有梯度（detach）"


def test_agg_loss_gradient_weights(m):
    # 对 loss_mat 求导，得到每个 token 在各模式下的权重；mask 掉的位置（含垃圾值）权重为 0
    M = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
    Mf = M.to(F64)
    n_tok = torch.tensor([3.0, 1.0, 1.0], dtype=F64)[:, None]  # 第 3 行全无效，下面乘 0，分母随便
    cases = {
        "token-mean": Mf / 4,
        "seq-mean-token-sum": Mf / 2,
        "seq-mean-token-mean": Mf / n_tok / 2,
        "seq-mean-token-sum-norm": Mf / 2 / 4,
    }
    for mode, w in cases.items():
        L = _t([[1.0, 2.0, 3.0, 7.0], [4.0, 5.0, 6.0, 7.0], [9.0, 9.0, 9.0, 9.0]]).requires_grad_(True)
        out = m.agg_loss(L, M, mode)
        assert out.dim() == 0, f"{mode}: agg_loss 应返回 0-d tensor"
        out.backward()
        _close(L.grad, w, atol=1e-7)
    L32 = torch.ones(2, 3, dtype=torch.float32)
    assert m.agg_loss(L32, torch.ones(2, 3, dtype=torch.bool), "seq-mean-token-mean").dtype == torch.float32, \
        "输出 dtype 应与 loss_mat 一致"
