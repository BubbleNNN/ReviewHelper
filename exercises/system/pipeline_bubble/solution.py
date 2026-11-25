def stage_order(schedule, p, m, stage):
    if schedule == "gpipe":
        return [("F", i) for i in range(m)] + [("B", i) for i in range(m)]
    if schedule == "1f1b":
        w = min(p - stage - 1, m)  # warmup 前向个数
        ops = [("F", i) for i in range(w)]
        for i in range(m - w):     # steady：一前一后
            ops += [("F", w + i), ("B", i)]
        ops += [("B", i) for i in range(m - w, m)]  # cooldown
        return ops
    raise ValueError(f"unknown schedule {schedule}")


def simulate(schedule, p, m, tf=1.0, tb=2.0):
    orders = [stage_order(schedule, p, m, s) for s in range(p)]
    end = {}                      # (op, stage, mb) -> 结束时间
    timeline = [[] for _ in range(p)]
    ptr = [0] * p
    free = [0.0] * p
    remaining = p * 2 * m
    while remaining:
        progress = False
        for s in range(p):
            if ptr[s] == len(orders[s]):
                continue
            op, i = orders[s][ptr[s]]
            if op == "F":
                deps = [("F", s - 1, i)] if s > 0 else []
            else:
                deps = [("F", s, i)] + ([("B", s + 1, i)] if s < p - 1 else [])
            if any(d not in end for d in deps):
                continue
            start = max([free[s]] + [end[d] for d in deps])
            dur = tf if op == "F" else tb
            end[(op, s, i)] = start + dur
            timeline[s].append((op, i, start, start + dur))
            free[s] = start + dur
            ptr[s] += 1
            remaining -= 1
            progress = True
        if not progress:
            raise RuntimeError("schedule deadlocked")
    total = max(free)
    busy = m * (tf + tb)
    peaks = []
    for s in range(p):
        cur = peak = 0
        for op, _ in orders[s]:
            cur += 1 if op == "F" else -1
            peak = max(peak, cur)
        peaks.append(peak)
    return {"timeline": timeline, "total_time": total, "bubble_ratio": 1 - busy / total, "peak_inflight": peaks}
