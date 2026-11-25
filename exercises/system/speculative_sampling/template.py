import numpy as np


def sample_from(probs, u):
    """逆 CDF 采样（已给出）：返回最小的 t 使得 cumsum(probs)[t] > u。probs 会先归一化。"""
    probs = np.asarray(probs, dtype=np.float64)
    c = np.cumsum(probs / probs.sum())
    t = int(np.searchsorted(c, u, side="right"))
    return min(t, len(probs) - 1)


def verify(p, q, draft_tokens, u_accept, u_sample):
    """链式投机采样的验证步。

    Args:
        p: (γ+1, V) target 在每个位置的分布（最后一行用于 bonus token）
        q: (γ, V)   draft 在每个位置的分布
        draft_tokens: 长度 γ，draft 采出的 token
        u_accept: 长度 γ 的 [0,1) 均匀随机数，第 k 个用于第 k 个 draft token 的接受判定
        u_sample: 一个 [0,1) 均匀随机数，用于最后那个 token（修正 token 或 bonus token）
    Returns:
        输出 token 列表：接受的 draft token + 1 个修正/bonus token，长度 1..γ+1
    """
    raise NotImplementedError
