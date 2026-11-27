import torch


def agg_loss(loss_mat, loss_mask, loss_agg_mode, loss_scale_factor=None):
    """把 (B, T) 的逐 token 损失聚合成标量（dp_size=1）。

    Args:
        loss_mat:  (B, T) float tensor，逐 token 损失（通常带梯度）
        loss_mask: (B, T) bool tensor，True=参与计算的 token
        loss_agg_mode: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean" /
            "seq-mean-token-sum-norm"，其他值抛 ValueError
        loss_scale_factor: seq-mean-token-sum-norm 的除数（python float），默认 T
    Returns:
        0-d tensor，dtype 与 loss_mat 相同，保留计算图（之后要 backward）。
        mask=False 的位置梯度为 0。
    """
    raise NotImplementedError


def ppo_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                    clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0,
                    loss_agg_mode="token-mean"):
    """PPO-clip 策略损失（含 dual-clip），对齐 verl compute_policy_loss_vanilla。

    Args:
        old_log_prob:  (B, T) float tensor，rollout 时的 log π_old，视为常数
        log_prob:      (B, T) float tensor，当前策略的 log π，带梯度
        advantages:    (B, T) float tensor，视为常数
        response_mask: (B, T) bool tensor
        clip_ratio_low / clip_ratio_high / clip_ratio_c: python float
    Returns:
        loss: 0-d tensor，dtype 与 log_prob 相同，梯度只流向 log_prob
              （old_log_prob、advantages 要 detach）
        metrics: {"pg_clipfrac": float, "ppo_kl": float, "pg_clipfrac_lower": float}，
                 都是 python float（用 .item()），只用于日志
    """
    raise NotImplementedError
