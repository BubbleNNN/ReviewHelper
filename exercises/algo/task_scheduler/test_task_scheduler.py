import random
from collections import Counter, deque


def _bfs_optimal(tasks, n):
    """在 (剩余次数, 冷却剩余) 状态空间上 BFS，求最短时间（只用于小输入）。"""
    kinds = sorted(set(tasks))
    cnt = Counter(tasks)
    start = (tuple(cnt[k] for k in kinds), tuple(0 for _ in kinds))
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (rem, cool), t = q.popleft()
        if sum(rem) == 0:
            return t
        nxt_cool = tuple(max(0, c - 1) for c in cool)
        moves = [(rem, nxt_cool)]  # idle
        for i in range(len(kinds)):
            if rem[i] > 0 and cool[i] == 0:
                r = list(rem)
                r[i] -= 1
                c = list(nxt_cool)
                c[i] = n
                moves.append((tuple(r), tuple(c)))
        for s in moves:
            if s not in seen:
                seen.add(s)
                q.append((s, t + 1))
    raise AssertionError("unreachable")


def _check_schedule(seq, tasks, n):
    assert Counter(x for x in seq if x != "idle") == Counter(tasks), "任务次数不对"
    assert not seq or seq[-1] != "idle", "末尾不能是 idle"
    last = {}
    for i, x in enumerate(seq):
        if x == "idle":
            continue
        if x in last:
            assert i - last[x] > n, f"{x} 在 {last[x]} 和 {i} 之间冷却不够"
        last[x] = i


def test_examples(m):
    assert m.least_interval(["A", "A", "A", "B", "B", "B"], 2) == 8
    assert m.least_interval(["A", "A", "A", "B", "B", "B"], 0) == 6
    assert m.least_interval(["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], 2) == 16


def test_no_idle_needed(m):
    assert m.least_interval(list("AABBCCDDEE"), 2) == 10
    assert m.least_interval(["A"], 100) == 1


def test_schedule_examples(m):
    for tasks, n in ((list("AAABBB"), 2), (list("AAAAAABCDEFG"), 2), (list("AABBCCDDEE"), 2), (["A"], 5)):
        seq = m.schedule(tasks, n)
        assert len(seq) == m.least_interval(tasks, n)
        _check_schedule(seq, tasks, n)


def test_least_interval_against_bfs(m):
    rng = random.Random(43)
    for _ in range(150):
        kinds = "ABC"[: rng.randint(1, 3)]
        tasks = [rng.choice(kinds) for _ in range(rng.randint(1, 7))]
        n = rng.randint(0, 3)
        assert m.least_interval(tasks, n) == _bfs_optimal(tasks, n), (tasks, n)


def test_schedule_random(m):
    rng = random.Random(47)
    for _ in range(300):
        kinds = "ABCDEF"[: rng.randint(1, 6)]
        tasks = [rng.choice(kinds) for _ in range(rng.randint(1, 30))]
        n = rng.randint(0, 5)
        seq = m.schedule(tasks, n)
        _check_schedule(seq, tasks, n)
        assert len(seq) == m.least_interval(tasks, n), (tasks, n, seq)
