import numpy as np


def compute_gae(rewards, values, mask, gamma=1.0, lam=1.0):
    """广义优势估计（GAE），mask=0 的位置被跳过。

    Args:
        rewards: (B, T) token 级奖励
        values:  (B, T) critic 估值
        mask:    (B, T) 1=策略生成的有效 token，0=padding / observation
        gamma, lam: 折扣因子与 GAE 的 λ
    Returns:
        advantages: (B, T)
        returns:    (B, T)，等于 advantages + values
    """
    raise NotImplementedError
