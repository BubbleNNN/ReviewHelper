import torch


def ring_allreduce(buffers):
    """模拟 ring all-reduce（reduce-scatter + all-gather）。

    Args:
        buffers: 长度为 p 的 list，每个元素是 (N,) 的 tensor（测试用 float64，也有 int64 的用例）
    Returns:
        results: 长度为 p 的 list，每个元素是 (N,) tensor，等于所有 buffers 的逐元素和，
                 dtype / device 与输入相同
        log: python list of (step, src, dst, chunk_id)，四个都是 python int，step 从 0 到 2(p-1)-1
    不要原地修改输入。
    """
    raise NotImplementedError


def allreduce_time(n_bytes, p, alpha, beta):
    """α-β 模型下 ring all-reduce 的耗时：2(p-1) 步，每步发 n_bytes/p 字节。

    Args:
        n_bytes: 每个 rank 上向量的总字节数（python float/int）
        p: rank 数（python int）；alpha: 每条消息的固定延迟（秒）；beta: 每字节耗时（秒/字节）
    Returns:
        python float，单位秒；p <= 1 时返回 0.0
    """
    raise NotImplementedError
