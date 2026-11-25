import numpy as np


def _groups(index):
    g = {}
    for i, key in enumerate(index):
        g.setdefault(key, []).append(i)
    return g


def grpo_advantage(token_level_rewards, response_mask, index, norm_adv_by_std=True, epsilon=1e-6):
    scores = np.asarray(token_level_rewards, dtype=np.float64).sum(axis=-1)
    adv = np.zeros_like(scores)
    for ids in _groups(index).values():
        s = scores[ids]
        if len(ids) == 1:
            mean, std = 0.0, 1.0
        else:
            mean, std = s.mean(), s.std(ddof=1)  # 无偏 std，与 torch.std 一致
        adv[ids] = (s - mean) / (std + epsilon) if norm_adv_by_std else s - mean
    return adv[:, None] * np.asarray(response_mask, dtype=np.float64)


def rloo_advantage(token_level_rewards, response_mask, index):
    scores = np.asarray(token_level_rewards, dtype=np.float64).sum(axis=-1)
    adv = scores.copy()
    for ids in _groups(index).values():
        n = len(ids)
        if n > 1:
            s = scores[ids]
            adv[ids] = s - (s.sum() - s) / (n - 1)
    return adv[:, None] * np.asarray(response_mask, dtype=np.float64)
