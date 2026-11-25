import numpy as np


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def top_k_filter(probs, k):
    probs = np.asarray(probs, dtype=np.float64)
    V = probs.shape[0]
    if k <= 0 or k >= V:
        return probs / probs.sum()
    kth = np.sort(probs)[-k]  # 第 k 大的概率
    out = np.where(probs >= kth, probs, 0.0)
    return out / out.sum()


def top_p_filter(probs, p):
    probs = np.asarray(probs, dtype=np.float64)
    if p >= 1.0:
        return probs / probs.sum()
    order = np.argsort(-probs, kind="stable")
    csum = np.cumsum(probs[order])
    n_keep = min(int(np.searchsorted(csum, p, side="left")) + 1, probs.shape[0])
    out = np.zeros_like(probs)
    keep = order[:n_keep]
    out[keep] = probs[keep]
    return out / out.sum()


def process_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    if temperature < 1e-5:
        out = np.zeros_like(logits)
        out[int(np.argmax(logits))] = 1.0
        return out
    probs = _softmax(logits / temperature)
    probs = top_k_filter(probs, top_k)
    probs = top_p_filter(probs, top_p)
    return probs


def sample(logits, rng, temperature=1.0, top_k=0, top_p=1.0):
    probs = process_logits(logits, temperature, top_k, top_p)
    return int(rng.choice(probs.shape[0], p=probs))
