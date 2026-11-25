import itertools


def prefix_survival(conf):
    out = []
    for cs in conf:
        acc, row = 1.0, []
        for c in cs:
            acc *= c
            row.append(acc)
        out.append(row)
    return out


def _theta(a, lengths, sps):
    R = len(a)
    B = R + sum(lengths)
    tau = R + sum(sum(a[r][:l]) for r, l in enumerate(lengths))
    return tau * sps(B)


def greedy_schedule(conf, sps):
    """DSpark Algorithm 1：按 a_{r,j} 全局降序贪心接纳，Θ 不再上升就停。"""
    a = prefix_survival(conf)
    R = len(a)
    cand = sorted(((a[r][j - 1], r, j) for r in range(R) for j in range(1, len(a[r]) + 1) if a[r][j - 1] > 0),
                  key=lambda e: (-e[0], e[1], e[2]))
    lengths = [0] * R
    B, tau = R, float(R)
    best, best_l = tau * sps(B), list(lengths)
    for val, r, j in cand:
        lengths[r] = j          # a 随 j 单调不增 + 同值按 j 升序 → 一定是前缀
        B += 1
        tau += val
        th = tau * sps(B)
        if th > best:
            best, best_l = th, list(lengths)
        else:
            break               # 第一次不上升就停（Θ 单峰时即全局最优）
    return best_l, best


def brute_force_schedule(conf, sps):
    a = prefix_survival(conf)
    best, best_l = None, None
    for lengths in itertools.product(*[range(len(row) + 1) for row in a]):
        th = _theta(a, lengths, sps)
        if best is None or th > best + 1e-15:
            best, best_l = th, list(lengths)
    return best_l, best
