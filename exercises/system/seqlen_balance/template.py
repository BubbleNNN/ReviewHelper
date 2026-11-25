def balanced_partition(seqlens, k, equal_size=True):
    """把下标分成 k 组，最小化最大组的长度和。

    equal_size=True 时每组条数相同（不能整除则 ValueError）。
    Returns: list of k lists（原下标）
    """
    raise NotImplementedError


def micro_batches_by_tokens(seqlens, max_token_len):
    """装箱：每个 micro-batch 总 token 数 <= max_token_len，个数尽量少。

    单条超长时 ValueError。Returns: list of lists（原下标）
    """
    raise NotImplementedError


def reverse_index(groups):
    """返回 rev，使得 outputs[rev[i]] 对应原第 i 条序列。"""
    raise NotImplementedError
