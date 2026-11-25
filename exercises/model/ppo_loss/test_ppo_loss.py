import math

import numpy as np


def _one(m, A, r, **kw):
    """单个 token 的损失：old=0，new=log r。"""
    loss, met = m.ppo_policy_loss(np.zeros((1, 1)), np.array([[math.log(r)]]), np.array([[A]]),
                                  np.ones((1, 1)), **kw)
    return loss, met


def test_clip_regions_positive_advantage(m):
    loss, met = _one(m, 1.0, 1.5)   # r 超过 1.2：被截到 -1.2
    np.testing.assert_allclose(loss, -1.2, atol=1e-12)
    assert met["pg_clipfrac"] == 1.0
    loss, met = _one(m, 1.0, 0.5)   # r<1 的好动作：不截断
    np.testing.assert_allclose(loss, -0.5, atol=1e-12)
    assert met["pg_clipfrac"] == 0.0
    loss, _ = _one(m, 2.0, 5.0)
    np.testing.assert_allclose(loss, -2.4, atol=1e-12)


def test_clip_regions_negative_advantage_and_dual_clip(m):
    loss, met = _one(m, -1.0, 0.5)  # 坏动作概率已经降够：截到 0.8
    np.testing.assert_allclose(loss, 0.8, atol=1e-12)
    assert met["pg_clipfrac"] == 1.0 and met["pg_clipfrac_lower"] == 0.0
    loss, met = _one(m, -1.0, 1.5)  # r 在 (1, c) 之间：正常梯度
    np.testing.assert_allclose(loss, 1.5, atol=1e-12)
    assert met["pg_clipfrac"] == 0.0 and met["pg_clipfrac_lower"] == 0.0
    loss, met = _one(m, -1.0, 5.0)  # r 超过 c=3：dual-clip 截到 3
    np.testing.assert_allclose(loss, 3.0, atol=1e-12)
    assert met["pg_clipfrac_lower"] == 1.0
    loss, _ = _one(m, -1.0, 5.0, clip_ratio_c=10.0)
    np.testing.assert_allclose(loss, 5.0, atol=1e-12)


def test_clip_higher(m):
    loss, _ = _one(m, 1.0, 1.25, clip_ratio_low=0.2, clip_ratio_high=0.28)
    np.testing.assert_allclose(loss, -1.25, atol=1e-12)
    loss, _ = _one(m, 1.0, 1.4, clip_ratio_low=0.2, clip_ratio_high=0.28)
    np.testing.assert_allclose(loss, -1.28, atol=1e-12)
    loss, _ = _one(m, -1.0, 0.75, clip_ratio_low=0.2, clip_ratio_high=0.28)
    np.testing.assert_allclose(loss, 0.8, atol=1e-12)


def test_agg_modes_by_hand(m):
    L = np.array([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0]])
    M = np.array([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=float)  # 第 3 条整行无效
    np.testing.assert_allclose(m.agg_loss(L, M, "token-mean"), 10 / 4, atol=1e-12)
    np.testing.assert_allclose(m.agg_loss(L, M, "seq-mean-token-sum"), (6 + 4) / 2, atol=1e-12)
    np.testing.assert_allclose(m.agg_loss(L, M, "seq-mean-token-mean"), (2 + 4) / 2, atol=1e-6)
    np.testing.assert_allclose(m.agg_loss(L, M, "seq-mean-token-sum-norm"), 5 / 4, atol=1e-12)
    np.testing.assert_allclose(m.agg_loss(L, M, "seq-mean-token-sum-norm", loss_scale_factor=10), 0.5, atol=1e-12)


def test_invalid_mode_raises(m):
    try:
        m.agg_loss(np.ones((1, 2)), np.ones((1, 2)), "bogus-mode")
    except ValueError:
        return
    raise AssertionError("非法 loss_agg_mode 应抛 ValueError")


def test_batch_metrics_and_mask(m):
    # 第 0 行：四种情况各一个 token；第 1 行：只有第 0 个 token 有效，其余是垃圾
    r = np.array([[1.5, 0.5, 0.5, 5.0], [1.0, 50.0, 1e-9, 7.0]])
    A = np.array([[1.0, 1.0, -1.0, -1.0], [2.0, -9.0, 9.0, -9.0]])
    M = np.array([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=float)
    loss, met = m.ppo_policy_loss(np.zeros((2, 4)), np.log(r), A, M)
    # 有效 token 损失：-1.2, -0.5, 0.8, 3.0, -2.0
    np.testing.assert_allclose(loss, (-1.2 - 0.5 + 0.8 + 3.0 - 2.0) / 5, atol=1e-12)
    # L2 > L1 的只有 (A=1, r=1.5) 和 (A=-1, r=0.5)；(A=-1, r=5) 是被 dual-clip 截的，算在 lower 里
    np.testing.assert_allclose(met["pg_clipfrac"], 2 / 5, atol=1e-12)
    np.testing.assert_allclose(met["pg_clipfrac_lower"], 1 / 5, atol=1e-12)
    np.testing.assert_allclose(met["ppo_kl"], -np.log(r[0]).sum() / 5, atol=1e-12)
    loss2, _ = m.ppo_policy_loss(np.zeros((2, 4)), np.log(r), A, M, loss_agg_mode="seq-mean-token-mean")
    np.testing.assert_allclose(loss2, ((-1.2 - 0.5 + 0.8 + 3.0) / 4 + (-2.0)) / 2, atol=1e-6)


def test_log_ratio_is_clamped(m):
    old = np.array([[0.0, 0.0]])
    new = np.array([[50.0, -50.0]])
    loss, met = m.ppo_policy_loss(old, new, np.array([[1.0, -1.0]]), np.ones((1, 2)))
    assert math.isfinite(loss)
    np.testing.assert_allclose(met["ppo_kl"], 0.0, atol=1e-12)  # (-20 + 20) / 2
    loss1, met1 = m.ppo_policy_loss(old[:, :1], new[:, :1], np.array([[1.0]]), np.ones((1, 1)))
    np.testing.assert_allclose(met1["ppo_kl"], -20.0, atol=1e-12)
    np.testing.assert_allclose(loss1, -1.2, atol=1e-12)
