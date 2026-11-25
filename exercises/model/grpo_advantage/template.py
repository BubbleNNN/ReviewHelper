import numpy as np


def grpo_advantage(token_level_rewards, response_mask, index, norm_adv_by_std=True, epsilon=1e-6):
    """GRPO 组内标准化优势（norm_adv_by_std=False 即 Dr.GRPO）。

    Args:
        token_level_rewards: (B, T)，score = 按行求和
        response_mask: (B, T)
        index: 长度 B，同组同值（可为字符串）
    Returns:
        (B, T) = 标量优势 × response_mask
    """
    raise NotImplementedError


def rloo_advantage(token_level_rewards, response_mask, index):
    """RLOO：A_i = s_i - 其余 n-1 条的均值；单条组 baseline 为 0。返回 (B, T)。"""
    raise NotImplementedError
