import math

import numpy as np


def _outcome(scores, T=3):
    """把标量分数放到最后一个 token 上。"""
    r = np.zeros((len(scores), T))
    r[:, -1] = scores
    return r


def test_grpo_hand_computed(m):
    # 分数 [1,0,1,0]：mean=0.5，无偏 std = sqrt(4·0.25/3) = sqrt(1/3)
    r = _outcome([1, 0, 1, 0])
    adv = m.grpo_advantage(r, np.ones((4, 3)), ["a", "a", "a", "a"])
    a = 0.5 / (math.sqrt(1 / 3) + 1e-6)
    np.testing.assert_allclose(adv[:, 0], [a, -a, a, -a], atol=1e-12)
    np.testing.assert_allclose(adv[:, 1], adv[:, 0])  # 广播到每个 token


def test_dr_grpo_no_std(m):
    r = _outcome([1, 0, 1, 0])
    adv = m.grpo_advantage(r, np.ones((4, 3)), [0, 0, 0, 0], norm_adv_by_std=False)
    np.testing.assert_allclose(adv[:, 0], [0.5, -0.5, 0.5, -0.5], atol=1e-12)


def test_interleaved_groups_and_degenerate(m):
    # 组 u1: 分数 3, 1（位置 0、2）；组 u2: 2, 2, 2（全相同）；组 u3：单条 5
    r = _outcome([3, 2, 1, 2, 5, 2])
    idx = np.array(["u1", "u2", "u1", "u2", "u3", "u2"])
    adv = m.grpo_advantage(r, np.ones((6, 3)), idx)
    s = math.sqrt(2.0)  # [3,1] 的无偏 std
    np.testing.assert_allclose(adv[[0, 2], 0], [1 / (s + 1e-6), -1 / (s + 1e-6)], atol=1e-12)
    np.testing.assert_allclose(adv[[1, 3, 5], 0], 0.0, atol=1e-12)  # 全对/全错组：优势为 0
    np.testing.assert_allclose(adv[4, 0], 5 / (1 + 1e-6), atol=1e-12)  # 单条组：mean 0, std 1


def test_response_mask_applied(m):
    r = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
    adv = m.grpo_advantage(r, mask, [7, 7], norm_adv_by_std=False)
    np.testing.assert_allclose(adv, [[0.5, 0.5, 0.5, 0.0], [-0.5, -0.5, 0.0, 0.0]], atol=1e-12)


def test_scores_are_row_sums(m):
    # 过程奖励分散在多个 token 上时，score 取每行之和
    r = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
    adv = m.grpo_advantage(r, np.ones((3, 3)), [1, 1, 1], norm_adv_by_std=False)
    np.testing.assert_allclose(adv[:, 0], np.array([1.0, 0.0, 1.0]) - 2 / 3, atol=1e-12)


def test_rloo_hand_and_identity(m):
    r = _outcome([1, 0, 0])
    adv = m.rloo_advantage(r, np.ones((3, 3)), ["p", "p", "p"])
    np.testing.assert_allclose(adv[:, 0], [1.0, -0.5, -0.5], atol=1e-12)
    rng = np.random.default_rng(0)
    s = rng.normal(size=5)
    a = m.rloo_advantage(_outcome(s), np.ones((5, 3)), [0] * 5)[:, 0]
    np.testing.assert_allclose(a, 5 / 4 * (s - s.mean()), atol=1e-12)  # = n/(n-1)·(s - mean)


def test_rloo_singleton_and_groups(m):
    r = _outcome([4, 1, 3, 9])
    adv = m.rloo_advantage(r, np.ones((4, 3)), ["x", "y", "x", "z"])
    np.testing.assert_allclose(adv[:, 0], [1.0, 1.0, -1.0, 9.0], atol=1e-12)
