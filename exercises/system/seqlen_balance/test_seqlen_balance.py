import numpy as np


def _lens(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(lo, hi, size=n)]


def _is_partition(groups, n):
    flat = sorted(i for g in groups for i in g)
    return flat == list(range(n))


def test_partition_valid_equal_size(m):
    L = _lens(48, 10, 3000, 1)
    groups = m.balanced_partition(L, 6)
    assert len(groups) == 6
    assert _is_partition(groups, 48)
    assert all(len(g) == 8 for g in groups)
    try:
        m.balanced_partition(L[:47], 6)
    except ValueError:
        pass
    else:
        raise AssertionError("47 条不能等分成 6 组，应抛 ValueError")


def test_partition_known_optimum(m):
    groups = m.balanced_partition([8, 7, 6, 5, 4, 3, 2, 1], 2)
    sums = sorted(sum([8, 7, 6, 5, 4, 3, 2, 1][i] for i in g) for g in groups)
    assert all(len(g) == 4 for g in groups)
    assert sums[-1] <= 19  # 最优是 18/18


def test_partition_beats_random(m):
    L = _lens(64, 50, 4000, 0)
    groups = m.balanced_partition(L, 8)
    mx = max(sum(L[i] for i in g) for g in groups)
    avg = sum(L) / 8
    assert mx <= 1.05 * avg
    rng = np.random.default_rng(123)
    for _ in range(20):
        perm = rng.permutation(64)
        rnd = max(sum(L[i] for i in perm[j * 8:(j + 1) * 8]) for j in range(8))
        assert mx <= rnd


def test_partition_unequal_bound(m):
    L = [1000, 3, 3, 3, 3, 900, 2, 2, 50, 40, 30]
    groups = m.balanced_partition(L, 3, equal_size=False)
    assert _is_partition(groups, len(L))
    mx = max(sum(L[i] for i in g) for g in groups)
    assert mx <= sum(L) / 3 + max(L)
    assert mx == 1000  # 最大那条单独一组即为最优


def test_micro_batches_budget_and_count(m):
    groups = m.micro_batches_by_tokens([6, 5, 4, 3, 2, 1], 7)
    assert _is_partition(groups, 6)
    assert len(groups) == 3  # 下界 ceil(21/7)=3
    for g in groups:
        assert sum([6, 5, 4, 3, 2, 1][i] for i in g) <= 7


def test_micro_batches_random(m):
    L = _lens(50, 1, 1000, 7)
    cap = 2048
    groups = m.micro_batches_by_tokens(L, cap)
    assert _is_partition(groups, 50)
    for g in groups:
        assert sum(L[i] for i in g) <= cap
    assert len(groups) <= 2 * sum(L) / cap + 1
    assert len(groups) <= int(np.ceil(sum(L) / cap)) + 2  # FFD 在这种分布下接近下界
    try:
        m.micro_batches_by_tokens([10, 5000], 4096)
    except ValueError:
        pass
    else:
        raise AssertionError("单条超过预算应抛 ValueError")


def test_reverse_index_restores_order(m):
    L = _lens(30, 1, 500, 9)
    groups = m.micro_batches_by_tokens(L, 900)
    outputs = []  # 模拟按 micro-batch 顺序跑模型，输出 = 长度 * 10
    for g in groups:
        outputs.extend(L[i] * 10 for i in g)
    rev = m.reverse_index(groups)
    assert [outputs[rev[i]] for i in range(30)] == [x * 10 for x in L]
    assert m.reverse_index([[2, 0], [1]]) == [1, 2, 0]
