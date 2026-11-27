import torch


def _masked_mean_item(x, mask):
    """日志用的 masked mean，返回 python float。"""
    x = x.detach().to(torch.float64)
    return (torch.where(mask, x, 0.0).sum() / mask.sum()).item()


def agg_loss(loss_mat, loss_mask, loss_agg_mode, loss_scale_factor=None):
    m = loss_mask.to(loss_mat.dtype)  # bool -> float，避免 long 和 1e-8 相加时落到 float32
    masked = torch.where(loss_mask, loss_mat, 0.0)  # mask 掉的位置前向为 0，反向梯度也为 0
    if loss_agg_mode == "token-mean":
        return masked.sum() / m.sum()
    tok_per_seq = m.sum(dim=-1)
    seq_valid = tok_per_seq > 0  # 整行被 mask 的序列不计数
    n_seq = seq_valid.sum()
    if loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        seq_losses = masked.sum(dim=-1)
        loss = torch.where(seq_valid, seq_losses, 0.0).sum() / n_seq
        if loss_agg_mode == "seq-mean-token-sum-norm":
            loss = loss / (loss_scale_factor if loss_scale_factor is not None else loss_mask.shape[-1])
        return loss
    if loss_agg_mode == "seq-mean-token-mean":
        seq_losses = masked.sum(dim=-1) / (tok_per_seq + 1e-8)
        return torch.where(seq_valid, seq_losses, 0.0).sum() / n_seq
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def ppo_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                    clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0,
                    loss_agg_mode="token-mean"):
    assert clip_ratio_c > 1.0
    old_log_prob = old_log_prob.detach()  # 旧策略和优势都是常数，梯度只流向 log_prob
    A = advantages.detach()
    neg_kl = torch.clamp(log_prob - old_log_prob, -20.0, 20.0)
    ratio = torch.exp(neg_kl)

    pg1 = -A * ratio
    pg2 = -A * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip1 = torch.maximum(pg1, pg2)
    pg3 = -A * clip_ratio_c
    clip2 = torch.minimum(pg3, clip1)
    pg_losses = torch.where(A < 0, clip2, clip1)

    metrics = {
        "pg_clipfrac": _masked_mean_item(pg2 > pg1, response_mask),
        "ppo_kl": _masked_mean_item(-neg_kl, response_mask),
        "pg_clipfrac_lower": _masked_mean_item((clip1 > pg3) & (A < 0), response_mask),
    }
    return agg_loss(pg_losses, response_mask, loss_agg_mode), metrics
