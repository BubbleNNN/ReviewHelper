import torch


@torch.no_grad()  # 优势和 returns 是常数：critic loss 拿 returns 当目标，不能反传进 values
def compute_gae(rewards, values, mask, gamma=1.0, lam=1.0):
    B, T = rewards.shape
    m_all = mask.to(rewards.dtype)  # bool 不能直接做 1 - m

    next_values = rewards.new_zeros(B)
    last_gae = rewards.new_zeros(B)
    adv = torch.zeros_like(rewards)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        gae_t = delta + gamma * lam * last_gae
        m = m_all[:, t]
        # mask=0 的位置对 next_values / last_gae 透明
        next_values = values[:, t] * m + (1 - m) * next_values
        last_gae = gae_t * m + (1 - m) * last_gae
        adv[:, t] = last_gae
    return adv, adv + values
