import numpy as np


def ring_allreduce(buffers):
    """模拟 ring all-reduce（reduce-scatter + all-gather）。

    Args:
        buffers: 长度为 p 的 list，每个元素 shape (N,)
    Returns:
        results: 长度为 p 的 list，每个元素都是所有 buffers 的逐元素和
        log: list of (step, src, dst, chunk_id)，step 从 0 到 2(p-1)-1
    """
    raise NotImplementedError


def allreduce_time(n_bytes, p, alpha, beta):
    """α-β 模型下 ring all-reduce 的耗时：2(p-1) 步，每步发 n_bytes/p 字节。"""
    raise NotImplementedError
