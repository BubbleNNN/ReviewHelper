import numpy as np


def layernorm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(axis=-1, keepdims=True)  # 有偏方差
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xc * rstd
    y = xhat * gamma + beta
    return y, (xhat, rstd, np.asarray(gamma, dtype=np.float64))


def layernorm_backward(dy, cache):
    xhat, rstd, gamma = cache
    dy = np.asarray(dy, dtype=np.float64)
    dgamma = (dy * xhat).sum(axis=0)
    dbeta = dy.sum(axis=0)
    g = dy * gamma  # 对 xhat 的梯度
    dx = rstd * (g - g.mean(axis=-1, keepdims=True) - xhat * (g * xhat).mean(axis=-1, keepdims=True))
    return dx, dgamma, dbeta


def rmsnorm(x, weight, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight
