def merge(intervals):
    out = []
    for s, e in sorted(intervals, key=lambda x: x[0]):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def insert(intervals, new):
    ns, ne = new
    out = []
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < ns:  # 完全在左边
        out.append(list(intervals[i]))
        i += 1
    while i < n and intervals[i][0] <= ne:  # 与 new 有交集
        ns = min(ns, intervals[i][0])
        ne = max(ne, intervals[i][1])
        i += 1
    out.append([ns, ne])
    while i < n:  # 完全在右边
        out.append(list(intervals[i]))
        i += 1
    return out
