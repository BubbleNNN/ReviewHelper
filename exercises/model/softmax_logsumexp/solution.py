import torch


def _safe_max(x, dim):
    m = torch.amax(x, dim=dim, keepdim=True)
    # 整行 -inf 时 m=-inf，x - m 会变成 -inf - (-inf) = nan；把非有限的 max 换成 0
    return torch.where(torch.isfinite(m), m, torch.zeros_like(m))


def logsumexp(x, dim=-1, keepdim=False):
    m = _safe_max(x, dim)
    out = torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True)) + m  # log(0) = -inf，不会报错
    return out if keepdim else out.squeeze(dim)


def softmax(x, dim=-1):
    e = torch.exp(x - _safe_max(x, dim))
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax(x, dim=-1):
    return x - logsumexp(x, dim=dim, keepdim=True)
