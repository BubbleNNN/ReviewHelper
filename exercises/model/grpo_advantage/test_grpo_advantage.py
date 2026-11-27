import math

import torch

F64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=F64)


def _close(actual, expected, atol=1e-12):
    torch.testing.assert_close(actual, expected if torch.is_tensor(expected) else _t(expected), atol=atol, rtol=0)


def _ones_mask(*shape):
    return torch.ones(*shape, dtype=torch.bool)


def _outcome(scores, T=3):
    """把标量分数放到最后一个 token 上。"""
    scores = torch.as_tensor(scores, dtype=F64)
    r = torch.zeros(len(scores), T, dtype=F64)
    r[:, -1] = scores
    return r


def test_grpo_hand_computed(m):
    # 分数 [1,0,1,0]：mean=0.5，无偏 std = sqrt(4·0.25/3) = sqrt(1/3)
    r = _outcome([1, 0, 1, 0])
    adv = m.grpo_advantage(r, _ones_mask(4, 3), ["a", "a", "a", "a"])
    a = 0.5 / (math.sqrt(1 / 3) + 1e-6)
    _close(adv[:, 0], [a, -a, a, -a])
    _close(adv[:, 1], adv[:, 0], atol=0)  # 广播到每个 token


def test_dr_grpo_no_std(m):
    r = _outcome([1, 0, 1, 0])
    adv = m.grpo_advantage(r, _ones_mask(4, 3), [0, 0, 0, 0], norm_adv_by_std=False)
    _close(adv[:, 0], [0.5, -0.5, 0.5, -0.5])


def test_interleaved_groups_and_degenerate(m):
    # 组 u1: 分数 3, 1（位置 0、2）；组 u2: 2, 2, 2（全相同）；组 u3：单条 5
    r = _outcome([3, 2, 1, 2, 5, 2])
    idx = ["u1", "u2", "u1", "u2", "u3", "u2"]
    adv = m.grpo_advantage(r, _ones_mask(6, 3), idx)
    s = math.sqrt(2.0)  # [3,1] 的无偏 std
    _close(adv[[0, 2], 0], [1 / (s + 1e-6), -1 / (s + 1e-6)])
    _close(adv[[1, 3, 5], 0], [0.0, 0.0, 0.0])  # 全对/全错组：优势为 0
    _close(adv[4, 0], 5 / (1 + 1e-6))  # 单条组：mean 0, std 1


def test_response_mask_applied(m):
    r = _t([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    adv = m.grpo_advantage(r, mask, [7, 7], norm_adv_by_std=False)
    _close(adv, [[0.5, 0.5, 0.5, 0.0], [-0.5, -0.5, 0.0, 0.0]])


def test_scores_are_row_sums(m):
    # 过程奖励分散在多个 token 上时，score 取每行之和
    r = _t([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
    adv = m.grpo_advantage(r, _ones_mask(3, 3), [1, 1, 1], norm_adv_by_std=False)
    _close(adv[:, 0], _t([1.0, 0.0, 1.0]) - 2 / 3)


def test_rloo_hand_and_identity(m):
    r = _outcome([1, 0, 0])
    adv = m.rloo_advantage(r, _ones_mask(3, 3), ["p", "p", "p"])
    _close(adv[:, 0], [1.0, -0.5, -0.5])
    g = torch.Generator().manual_seed(0)
    s = torch.randn(5, generator=g, dtype=F64)
    a = m.rloo_advantage(_outcome(s), _ones_mask(5, 3), [0] * 5)[:, 0]
    _close(a, 5 / 4 * (s - s.mean()))  # = n/(n-1)·(s - mean)


def test_rloo_singleton_and_groups(m):
    r = _outcome([4, 1, 3, 9])
    adv = m.rloo_advantage(r, _ones_mask(4, 3), ["x", "y", "x", "z"])
    _close(adv[:, 0], [1.0, 1.0, -1.0, 9.0])


def test_tensor_index(m):
    # index 是 long tensor 时也要正确分组（tensor 元素按 id 哈希，要先 .tolist()）
    r = _outcome([1, 5, 3, 5])
    idx = torch.tensor([0, 1, 0, 1])
    _close(m.grpo_advantage(r, _ones_mask(4, 3), idx, norm_adv_by_std=False)[:, 0], [-1.0, 0.0, 1.0, 0.0])
    _close(m.rloo_advantage(r, _ones_mask(4, 3), idx)[:, 0], [-2.0, 0.0, 2.0, 0.0])


def test_no_grad_and_dtype(m):
    # 优势是常数：即使 reward 带梯度，输出也不该带；dtype 跟随输入
    r = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 3.0]], dtype=torch.float32, requires_grad=True)
    mask = _ones_mask(3, 2)
    for name, fn in (("grpo_advantage", m.grpo_advantage), ("rloo_advantage", m.rloo_advantage)):
        adv = fn(r, mask, ["a", "a", "a"])
        assert adv.dtype == torch.float32, f"{name}: 输出 dtype 应与输入一致"
        assert not adv.requires_grad, f"{name}: 优势不应带梯度（用 torch.no_grad 或 detach）"
        assert adv.shape == (3, 2)
