import torch


def compute_gae(rewards, values, mask, gamma=1.0, lam=1.0):
    """广义优势估计（GAE），mask=False 的位置被跳过。

    Args:
        rewards: (B, T) float tensor，token 级奖励
        values:  (B, T) float tensor，critic 估值（可能带梯度）
        mask:    (B, T) bool tensor，True=策略生成的有效 token，False=padding / observation
        gamma, lam: 折扣因子与 GAE 的 λ（python float）
    Returns:
        advantages: (B, T)，dtype 与 rewards 相同
        returns:    (B, T)，等于 advantages + values
        两者都不带梯度（requires_grad=False）。
    """
    raise NotImplementedError
