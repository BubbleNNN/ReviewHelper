import numpy as np


def flash_attention(q, k, v, causal=False, block_q=16, block_k=16):
    """分块 + online softmax 的注意力前向（单头）。

    Args:
        q: (Tq, d)
        k: (Tk, d)
        v: (Tk, dv)
        causal: True 时 Tq == Tk，query i 只看 key j <= i
        block_q, block_k: 块大小，长度不一定能整除
    Returns:
        out: (Tq, dv)
        lse: (Tq,) 每行缩放后分数的 logsumexp
    """
    raise NotImplementedError
