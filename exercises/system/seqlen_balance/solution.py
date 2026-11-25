def balanced_partition(seqlens, k, equal_size=True):
    """LPT 贪心：按长度降序，每次放进当前总长最小（且未满）的组。"""
    n = len(seqlens)
    if equal_size and n % k != 0:
        raise ValueError("equal_size=True 要求 len(seqlens) 能被 k 整除")
    cap = n // k if equal_size else n
    groups = [[] for _ in range(k)]
    sums = [0] * k
    for i in sorted(range(n), key=lambda i: (-seqlens[i], i)):
        g = min((j for j in range(k) if len(groups[j]) < cap), key=lambda j: (sums[j], j))
        groups[g].append(i)
        sums[g] += seqlens[i]
    return [sorted(g) for g in groups]


def micro_batches_by_tokens(seqlens, max_token_len):
    """First-Fit Decreasing：每个 micro-batch 的总 token 数不超过 max_token_len。"""
    if any(L > max_token_len for L in seqlens):
        raise ValueError("存在单条序列超过 max_token_len")
    bins, loads = [], []
    for i in sorted(range(len(seqlens)), key=lambda i: (-seqlens[i], i)):
        for b in range(len(bins)):
            if loads[b] + seqlens[i] <= max_token_len:
                bins[b].append(i)
                loads[b] += seqlens[i]
                break
        else:
            bins.append([i])
            loads.append(seqlens[i])
    return [sorted(b) for b in bins]


def reverse_index(groups):
    order = [i for g in groups for i in g]
    rev = [0] * len(order)
    for pos, i in enumerate(order):
        rev[i] = pos
    return rev
