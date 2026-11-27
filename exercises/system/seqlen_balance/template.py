def balanced_partition(seqlens, k, equal_size=True):
    """把下标分成 k 组，最小化最大组的长度和。

    equal_size=True 时每组条数相同（不能整除则 ValueError）。
    Args: seqlens: python list[int]；k: python int
    Returns: 长度为 k 的 python list，每个元素是 list[int]（原下标）
    """
    raise NotImplementedError


def micro_batches_by_tokens(seqlens, max_token_len):
    """装箱：每个 micro-batch 总 token 数 <= max_token_len，个数尽量少。

    单条超长时 ValueError。
    Args: seqlens: python list[int]；max_token_len: python int
    Returns: python list，每个元素是 list[int]（原下标）
    """
    raise NotImplementedError


def reverse_index(groups):
    """返回 rev（python list[int]），使得 outputs[rev[i]] 对应原第 i 条序列。

    Args: groups: list[list[int]]，按顺序执行的各组原下标
    """
    raise NotImplementedError
