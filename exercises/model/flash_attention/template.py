import torch


def flash_attention(q, k, v, causal=False, block_q=16, block_k=16):
    """分块 + online softmax 的注意力前向（单头）。

    不要调用 F.scaled_dot_product_attention，也不要构造完整的 (Tq, Tk) 分数矩阵。

    Args:
        q: (Tq, d) float tensor
        k: (Tk, d) float tensor
        v: (Tk, dv) float tensor
        causal: True 时 Tq == Tk，query i 只看 key j <= i
        block_q, block_k: python int，块大小，长度不一定能整除
    Returns:
        out: (Tq, dv) tensor
        lse: (Tq,) tensor，每行缩放后分数的 logsumexp
        dtype 都与 q 相同。
    """
    raise NotImplementedError
