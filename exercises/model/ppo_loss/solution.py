import numpy as np


def _masked_mean(x, mask):
    return float(np.where(mask > 0, x, 0.0).sum() / mask.sum())


def agg_loss(loss_mat, loss_mask, loss_agg_mode, loss_scale_factor=None):
    loss_mat = np.asarray(loss_mat, dtype=np.float64)
    mask = np.asarray(loss_mask, dtype=np.float64)
    masked = np.where(mask > 0, loss_mat, 0.0)
    if loss_agg_mode == "token-mean":
        return float(masked.sum() / mask.sum())
    tok_per_seq = mask.sum(axis=-1)
    seq_valid = tok_per_seq > 0  # 整行被 mask 的序列不计数
    if loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        seq_losses = masked.sum(axis=-1)
        loss = seq_losses[seq_valid].sum() / seq_valid.sum()
        if loss_agg_mode == "seq-mean-token-sum-norm":
            loss /= loss_scale_factor if loss_scale_factor is not None else mask.shape[-1]
        return float(loss)
    if loss_agg_mode == "seq-mean-token-mean":
        seq_losses = masked.sum(axis=-1) / (tok_per_seq + 1e-8)
        return float(seq_losses[seq_valid].sum() / seq_valid.sum())
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def ppo_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                    clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0,
                    loss_agg_mode="token-mean"):
    assert clip_ratio_c > 1.0
    old_log_prob, log_prob, A, mask = (np.asarray(a, dtype=np.float64)
                                       for a in (old_log_prob, log_prob, advantages, response_mask))
    neg_kl = np.clip(log_prob - old_log_prob, -20.0, 20.0)
    ratio = np.exp(neg_kl)

    pg1 = -A * ratio
    pg2 = -A * np.clip(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip1 = np.maximum(pg1, pg2)
    pg3 = -A * clip_ratio_c
    clip2 = np.minimum(pg3, clip1)
    pg_losses = np.where(A < 0, clip2, clip1)

    metrics = {
        "pg_clipfrac": _masked_mean((pg2 > pg1).astype(float), mask),
        "ppo_kl": _masked_mean(-neg_kl, mask),
        "pg_clipfrac_lower": _masked_mean(((clip1 > pg3) & (A < 0)).astype(float), mask),
    }
    return agg_loss(pg_losses, mask, loss_agg_mode), metrics
