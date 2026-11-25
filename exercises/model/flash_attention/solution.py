import numpy as np


def flash_attention(q, k, v, causal=False, block_q=16, block_k=16):
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    Tq, d = q.shape
    Tk, dv = v.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((Tq, dv))
    lse = np.zeros(Tq)

    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[qs:qe] * scale
        m_i = np.full(qe - qs, -np.inf)   # 行最大值
        l_i = np.zeros(qe - qs)           # 分母
        acc = np.zeros((qe - qs, dv))     # 未归一化输出

        for ks in range(0, Tk, block_k):
            if causal and ks > qe - 1:
                break  # 这个 key 块完全在未来，后面的也是
            ke = min(ks + block_k, Tk)
            s = qb @ k[ks:ke].T
            if causal:
                rows = np.arange(qs, qe)[:, None]
                cols = np.arange(ks, ke)[None, :]
                s = np.where(cols <= rows, s, -np.inf)
            m_new = np.maximum(m_i, s.max(axis=1))
            m_safe = np.where(np.isfinite(m_new), m_new, 0.0)
            p = np.exp(s - m_safe[:, None])
            alpha = np.exp(np.where(np.isfinite(m_i), m_i - m_safe, -np.inf))  # 旧基准换算到新基准
            l_i = alpha * l_i + p.sum(axis=1)
            acc = alpha[:, None] * acc + p @ v[ks:ke]
            m_i = m_new

        out[qs:qe] = acc / l_i[:, None]
        lse[qs:qe] = m_i + np.log(l_i)
    return out, lse
