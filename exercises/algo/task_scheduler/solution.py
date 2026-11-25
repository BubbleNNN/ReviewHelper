import heapq
from collections import Counter, deque


def least_interval(tasks, n):
    if not tasks:
        return 0
    cnt = Counter(tasks)
    maxc = max(cnt.values())
    cnt_max = sum(1 for v in cnt.values() if v == maxc)
    return max(len(tasks), (maxc - 1) * (n + 1) + cnt_max)


def schedule(tasks, n):
    ready = [(-c, t) for t, c in Counter(tasks).items()]  # 最大堆：剩余次数最多的优先
    heapq.heapify(ready)
    cooling = deque()  # (可再次执行的时刻, -剩余次数, 任务)
    out, time = [], 0
    while ready or cooling:
        while cooling and cooling[0][0] <= time:
            _, negc, t = cooling.popleft()
            heapq.heappush(ready, (negc, t))
        if ready:
            negc, t = heapq.heappop(ready)
            out.append(t)
            if negc + 1 < 0:
                cooling.append((time + n + 1, negc + 1, t))
        else:
            out.append("idle")
        time += 1
    return out
