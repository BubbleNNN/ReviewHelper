import math

import torch


def flash_attention(q, k, v, causal=False, block_q=16, block_k=16):
    Tq, d = q.shape
    Tk, dv = v.shape
    scale = 1.0 / math.sqrt(d)
    out = q.new_zeros(Tq, dv)
    lse = q.new_zeros(Tq)

    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[qs:qe] * scale
        m_i = q.new_full((qe - qs,), -math.inf)  # 行最大值
        l_i = q.new_zeros(qe - qs)               # 分母
        acc = q.new_zeros(qe - qs, dv)           # 未归一化输出

        for ks in range(0, Tk, block_k):
            if causal and ks > qe - 1:
                break  # 这个 key 块完全在未来，后面的也是
            ke = min(ks + block_k, Tk)
            s = qb @ k[ks:ke].T
            if causal:
                rows = torch.arange(qs, qe, device=q.device)[:, None]
                cols = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(cols > rows, -math.inf)
            m_new = torch.maximum(m_i, s.amax(dim=1))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))  # 整行被 mask 时防 nan
            p = torch.exp(s - m_safe[:, None])
            alpha = torch.exp(m_i - m_safe)  # 旧基准换算到新基准；m_i=-inf 时为 0
            l_i = alpha * l_i + p.sum(dim=1)
            acc = alpha[:, None] * acc + p @ v[ks:ke]
            m_i = m_new

        out[qs:qe] = acc / l_i[:, None]
        lse[qs:qe] = m_i + torch.log(l_i)
    return out, lse
