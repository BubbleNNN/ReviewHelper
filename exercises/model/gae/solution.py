import numpy as np


def compute_gae(rewards, values, mask, gamma=1.0, lam=1.0):
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    B, T = rewards.shape

    next_values = np.zeros(B)
    last_gae = np.zeros(B)
    adv = np.zeros((B, T))
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        gae_t = delta + gamma * lam * last_gae
        m = mask[:, t]
        # mask=0 的位置对 next_values / last_gae 透明
        next_values = values[:, t] * m + (1 - m) * next_values
        last_gae = gae_t * m + (1 - m) * last_gae
        adv[:, t] = last_gae
    return adv, adv + values
