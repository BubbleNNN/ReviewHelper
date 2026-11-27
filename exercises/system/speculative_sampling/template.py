import torch


def sample_from(probs, u):
    """逆 CDF 采样（已给出）：返回最小的 t 使得 cumsum(probs)[t] > u。probs 会先归一化。

    Args:
        probs: (V,) float tensor，非负即可（不必已归一化）
        u: [0,1) 的 python float 或 0 维 tensor
    Returns:
        python int，token id
    """
    c = torch.cumsum(probs / probs.sum(), dim=-1)
    u = torch.as_tensor(u, dtype=c.dtype, device=c.device).reshape(1)
    t = int(torch.searchsorted(c, u, right=True).item())
    return min(t, probs.shape[-1] - 1)


def verify(p, q, draft_tokens, u_accept, u_sample):
    """链式投机采样的验证步。

    Args:
        p: (γ+1, V) float tensor，target 在每个位置的分布（最后一行用于 bonus token）
        q: (γ, V)   float tensor，draft 在每个位置的分布，dtype 与 p 相同
        draft_tokens: (γ,) long tensor，draft 采出的 token
        u_accept: (γ,) float tensor，[0,1) 均匀随机数，第 k 个用于第 k 个 draft token 的接受判定
        u_sample: [0,1) 的 python float，用于最后那个 token（修正 token 或 bonus token）
    Returns:
        python list[int]：接受的 draft token + 1 个修正/bonus token，长度 1..γ+1
    """
    raise NotImplementedError
