import numpy as np


def sample_from(probs, u):
    """逆 CDF 采样：返回最小的 t 使得 cumsum(probs)[t] > u。"""
    probs = np.asarray(probs, dtype=np.float64)
    c = np.cumsum(probs / probs.sum())
    t = int(np.searchsorted(c, u, side="right"))
    return min(t, len(probs) - 1)


def verify(p, q, draft_tokens, u_accept, u_sample):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    gamma = len(draft_tokens)
    out = []
    for k in range(gamma):
        x = draft_tokens[k]
        if u_accept[k] < p[k, x] / q[k, x]:   # 以 min(1, p/q) 的概率接受
            out.append(int(x))
            continue
        residual = np.maximum(p[k] - q[k], 0.0)  # 拒绝：从 norm(max(0, p-q)) 重采样
        out.append(sample_from(residual, u_sample))
        return out
    out.append(sample_from(p[gamma], u_sample))  # 全部接受：从 p[γ] 采 bonus token
    return out
