import itertools
import random


def _brute(nums, k):
    """枚举所有 k-1 个切点。"""
    n = len(nums)
    best = None
    for cuts in itertools.combinations(range(1, n), k - 1):
        bounds = (0,) + cuts + (n,)
        worst = max(sum(nums[a:b]) for a, b in zip(bounds, bounds[1:]))
        best = worst if best is None else min(best, worst)
    return best


def _check_partition(nums, k, parts, opt):
    assert len(parts) == k, f"要恰好 {k} 段，得到 {len(parts)} 段"
    assert all(len(p) > 0 for p in parts), "每段必须非空"
    assert [x for p in parts for x in p] == list(nums), "拼起来要等于原数组"
    assert max(sum(p) for p in parts) == opt


def test_examples(m):
    assert m.split_array([7, 2, 5, 10, 8], 2) == 18
    assert m.split_array([1, 2, 3, 4, 5], 2) == 9
    assert m.split_array([1, 4, 4], 3) == 4


def test_extremes(m):
    nums = [3, 1, 4, 1, 5]
    assert m.split_array(nums, 1) == 14
    assert m.split_array(nums, 5) == 5
    assert m.split_array([0, 0, 0], 2) == 0


def test_partition_examples(m):
    for nums, k in ([7, 2, 5, 10, 8], 2), ([1, 4, 4], 3), ([0, 0, 0], 3), ([5], 1), ([2, 0, 0, 0, 9], 4):
        _check_partition(nums, k, m.partition(nums, k), _brute(nums, k))


def test_random_against_brute_force(m):
    rng = random.Random(13)
    for _ in range(300):
        n = rng.randint(1, 9)
        nums = [rng.randint(0, 12) for _ in range(n)]
        k = rng.randint(1, n)
        opt = _brute(nums, k)
        assert m.split_array(nums, k) == opt, (nums, k)
        _check_partition(nums, k, m.partition(nums, k), opt)


def test_large_layers(m):
    # 96 层，每层耗时 1~9，切成 8 个 stage
    rng = random.Random(2)
    nums = [rng.randint(1, 9) for _ in range(96)]
    opt = m.split_array(nums, 8)
    assert sum(nums) / 8 <= opt <= sum(nums) / 8 + max(nums)
    _check_partition(nums, 8, m.partition(nums, 8), opt)
