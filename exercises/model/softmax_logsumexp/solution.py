import numpy as np


def logsumexp(x, axis=-1, keepdims=False):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)  # 整行 -inf 时避免 -inf - (-inf) = nan
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)) + m_safe
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m_safe)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    return x - logsumexp(x, axis=axis, keepdims=True)
