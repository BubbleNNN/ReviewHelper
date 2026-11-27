import torch


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    var = (xc ** 2).mean(dim=-1, keepdim=True)  # 有偏方差（除以 D）
    rstd = torch.rsqrt(var + eps)
    xhat = xc * rstd
    y = xhat * gamma + beta
    return y, (xhat, rstd, gamma)


def layernorm_backward(dy, cache):
    xhat, rstd, gamma = cache
    dgamma = (dy * xhat).sum(dim=0)
    dbeta = dy.sum(dim=0)
    g = dy * gamma  # 对 xhat 的梯度
    dx = rstd * (g - g.mean(dim=-1, keepdim=True) - xhat * (g * xhat).mean(dim=-1, keepdim=True))
    return dx, dgamma, dbeta


def rmsnorm(x, weight, eps=1e-6):
    ms = (x ** 2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * weight
