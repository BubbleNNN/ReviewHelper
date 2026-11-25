import numpy as np


def agg_loss(loss_mat, loss_mask, loss_agg_mode, loss_scale_factor=None):
    """把 (B, T) 的逐 token 损失聚合成标量（dp_size=1）。

    loss_agg_mode ∈ {"token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"}，
    其他值抛 ValueError。seq-mean-token-sum-norm 的 loss_scale_factor 默认为 T。
    Returns: float
    """
    raise NotImplementedError


def ppo_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                    clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0,
                    loss_agg_mode="token-mean"):
    """PPO-clip 策略损失（含 dual-clip），对齐 verl compute_policy_loss_vanilla。

    Args:
        old_log_prob, log_prob, advantages, response_mask: (B, T)
    Returns:
        loss: float
        metrics: {"pg_clipfrac": float, "ppo_kl": float, "pg_clipfrac_lower": float}
    """
    raise NotImplementedError
