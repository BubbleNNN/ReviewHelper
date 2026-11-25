import random
from collections import Counter


def _brute(nums, k):
    ranked = sorted(Counter(nums).items(), key=lambda kv: (-kv[1], kv[0]))
    return [x for x, _ in ranked[:k]]


def _both(m):
    return (m.top_k_heap, m.top_k_bucket)


def test_example(m):
    for f in _both(m):
        assert f([1, 1, 1, 2, 2, 3], 2) == [1, 2]


def test_tie_break_by_value(m):
    for f in _both(m):
        assert f([4, 4, 1, 1, 2], 2) == [1, 4]
        assert f([5, 3, 9], 3) == [3, 5, 9]


def test_k_zero_and_all(m):
    for f in _both(m):
        assert f([1, 2, 2], 0) == []
        assert f([7, 7, 8, 9, 9, 9], 3) == [9, 7, 8]


def test_negative_numbers(m):
    for f in _both(m):
        assert f([-1, -1, -2, 3, 3, 3, -2, -2], 2) == [-2, 3]


def test_does_not_modify_input(m):
    nums = [3, 1, 2, 3, 1, 3]
    snapshot = list(nums)
    for f in _both(m):
        f(nums, 2)
    assert nums == snapshot


def test_random_against_brute_force(m):
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 60)
        nums = [rng.randint(-8, 8) for _ in range(n)]
        u = len(set(nums))
        k = rng.randint(0, u)
        want = _brute(nums, k)
        assert m.top_k_heap(nums, k) == want
        assert m.top_k_bucket(nums, k) == want


def test_large_input(m):
    rng = random.Random(11)
    nums = [int(rng.paretovariate(1.2)) for _ in range(50000)]
    want = _brute(nums, 20)
    assert m.top_k_heap(nums, 20) == want
    assert m.top_k_bucket(nums, 20) == want
