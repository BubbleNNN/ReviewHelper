import torch


def sample_from(probs, u):
    """逆 CDF 采样：返回最小的 t 使得 cumsum(probs)[t] > u（python int）。"""
    c = torch.cumsum(probs / probs.sum(), dim=-1)
    u = torch.as_tensor(u, dtype=c.dtype, device=c.device).reshape(1)
    t = int(torch.searchsorted(c, u, right=True).item())
    return min(t, probs.shape[-1] - 1)


def verify(p, q, draft_tokens, u_accept, u_sample):
    gamma = len(draft_tokens)
    out = []
    for k in range(gamma):
        x = int(draft_tokens[k])
        if u_accept[k] < p[k, x] / q[k, x]:   # 以 min(1, p/q) 的概率接受
            out.append(x)
            continue
        residual = torch.clamp(p[k] - q[k], min=0.0)  # 拒绝：从 norm(max(0, p-q)) 重采样
        out.append(sample_from(residual, u_sample))
        return out
    out.append(sample_from(p[gamma], u_sample))  # 全部接受：从 p[γ] 采 bonus token
    return out
