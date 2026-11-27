import torch


def grpo_advantage(token_level_rewards, response_mask, index, norm_adv_by_std=True, epsilon=1e-6):
    """GRPO 组内标准化优势（norm_adv_by_std=False 即 Dr.GRPO）。

    Args:
        token_level_rewards: (B, T) float tensor，score = 按行求和
        response_mask: (B, T) bool tensor
        index: 长度 B 的 python list（元素可哈希，如字符串 uid、int），同组同值；
            也可能是 (B,) long tensor，此时先 .tolist()（tensor 元素按对象 id 哈希，不能直接当 dict 的 key）
        norm_adv_by_std, epsilon: python bool / float
    Returns:
        (B, T) tensor = 标量优势 × response_mask，dtype 与 token_level_rewards 相同，不带梯度。
    """
    raise NotImplementedError


def rloo_advantage(token_level_rewards, response_mask, index):
    """RLOO：A_i = s_i - 其余 n-1 条的均值；单条组 baseline 为 0。

    参数含义同 grpo_advantage。返回 (B, T) tensor，dtype 与输入相同，不带梯度。
    """
    raise NotImplementedError
