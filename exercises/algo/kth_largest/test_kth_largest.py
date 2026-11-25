import random
import time


def _both(m):
    return (m.kth_largest_quickselect, m.kth_largest_heap)


def test_examples(m):
    for f in _both(m):
        assert f([3, 2, 1, 5, 6, 4], 2) == 5
        assert f([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4


def test_k_extremes(m):
    nums = [9, -1, 4, 4, 7]
    for f in _both(m):
        assert f(nums, 1) == 9
        assert f(nums, 5) == -1
        assert f([42], 1) == 42


def test_does_not_modify_input(m):
    nums = [5, 1, 4, 2, 3]
    snap = list(nums)
    for f in _both(m):
        f(nums, 3)
    assert nums == snap


def test_random_against_sorted(m):
    rng = random.Random(23)
    for _ in range(300):
        n = rng.randint(1, 50)
        nums = [rng.randint(-20, 20) for _ in range(n)]
        k = rng.randint(1, n)
        want = sorted(nums, reverse=True)[k - 1]
        for f in _both(m):
            assert f(nums, k) == want


def test_large_random_and_sorted(m):
    rng = random.Random(29)
    big = [rng.random() for _ in range(50000)]
    srt = sorted(big)
    for nums in (big, srt, srt[::-1]):
        for k in (1, 777, 25000, 50000):
            assert m.kth_largest_quickselect(nums, k) == srt[-k]
    assert m.kth_largest_heap(big, 777) == srt[-777]


def test_all_equal_is_fast(m):
    nums = [7] * 20000
    t0 = time.perf_counter()
    for k in (1, 20000):
        assert m.kth_largest_quickselect(nums, k) == 7
    dt = time.perf_counter() - t0
    assert dt < 1.0, f"全相同元素用了 {dt:.2f}s：划分退化成 O(n^2) 了，试试三路划分"
