import copy
import random


def _brute_union(intervals):
    """在“端点乘 2”的整数网格上涂色，再找连续段。闭区间 [a,b] -> 网格 2a..2b。"""
    if not intervals:
        return []
    lo = min(s for s, _ in intervals) * 2
    hi = max(e for _, e in intervals) * 2
    covered = [False] * (hi - lo + 1)
    for s, e in intervals:
        for g in range(2 * s, 2 * e + 1):
            covered[g - lo] = True
    out, start = [], None
    for i, c in enumerate(covered + [False]):
        g = i + lo
        if c and start is None:
            start = g
        if not c and start is not None:
            out.append([start // 2, (g - 1) // 2])
            start = None
    return out


def test_merge_example(m):
    assert m.merge([[1, 3], [8, 10], [2, 6], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]


def test_merge_touching_and_contained(m):
    assert m.merge([[3, 5], [1, 3]]) == [[1, 5]]
    assert m.merge([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    assert m.merge([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
    assert m.merge([[2, 2], [2, 2], [5, 5]]) == [[2, 2], [5, 5]]
    assert m.merge([]) == []


def test_insert_examples(m):
    assert m.insert([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]
    assert m.insert([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]) == [[1, 2], [3, 10], [12, 16]]
    assert m.insert([], [5, 7]) == [[5, 7]]
    assert m.insert([[1, 5]], [6, 8]) == [[1, 5], [6, 8]]
    assert m.insert([[3, 5]], [0, 1]) == [[0, 1], [3, 5]]
    assert m.insert([[1, 5]], [5, 7]) == [[1, 7]]


def test_does_not_modify_input(m):
    a = [[1, 4], [2, 3], [0, 1]]
    b = [[1, 2], [5, 6]]
    a0, b0 = copy.deepcopy(a), copy.deepcopy(b)
    m.merge(a)
    m.insert(b, [2, 5])
    assert a == a0 and b == b0


def test_merge_random_against_grid(m):
    rng = random.Random(3)
    for _ in range(400):
        ivs = []
        for _ in range(rng.randint(1, 12)):
            s = rng.randint(-10, 20)
            ivs.append([s, s + rng.randint(0, 6)])
        assert m.merge(ivs) == _brute_union(ivs)


def test_insert_random_against_grid(m):
    rng = random.Random(5)
    for _ in range(400):
        raw = []
        for _ in range(rng.randint(0, 8)):
            s = rng.randint(0, 40)
            raw.append([s, s + rng.randint(0, 4)])
        base = _brute_union(raw)  # 规范化：有序且不相交
        s = rng.randint(-2, 44)
        new = [s, s + rng.randint(0, 10)]
        assert m.insert(base, new) == _brute_union(base + [new])
