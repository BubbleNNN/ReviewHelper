import random
import time


def _brute(nums, k):
    return [max(nums[i:i + k]) for i in range(len(nums) - k + 1)]


def test_example(m):
    assert m.max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]


def test_k_one_and_k_n(m):
    nums = [4, -2, 9, 9, 0]
    assert m.max_sliding_window(nums, 1) == nums
    assert m.max_sliding_window(nums, 5) == [9]
    assert m.max_sliding_window([7], 1) == [7]


def test_monotonic_inputs(m):
    inc = list(range(10))
    dec = list(range(10, 0, -1))
    assert m.max_sliding_window(inc, 4) == list(range(3, 10))
    assert m.max_sliding_window(dec, 4) == list(range(10, 3, -1))


def test_duplicates(m):
    assert m.max_sliding_window([2, 2, 2, 1, 1, 2], 2) == [2, 2, 2, 1, 2]


def test_random_against_brute_force(m):
    rng = random.Random(9)
    for _ in range(400):
        n = rng.randint(1, 40)
        nums = [rng.randint(-5, 5) for _ in range(n)]
        k = rng.randint(1, n)
        assert m.max_sliding_window(nums, k) == _brute(nums, k)


def test_linear_time(m):
    # O(nk) 的写法在这里会明显变慢（约 5 亿次比较）；O(n) 写法只要几十毫秒
    rng = random.Random(1)
    n, k = 60000, 10000
    nums = [rng.random() for _ in range(n)]
    t0 = time.perf_counter()
    out = m.max_sliding_window(nums, k)
    dt = time.perf_counter() - t0
    assert len(out) == n - k + 1
    for i in (0, 12345, n - k):
        assert out[i] == max(nums[i:i + k])
    assert dt < 1.5, f"用时 {dt:.2f}s，看起来不是 O(n)"
