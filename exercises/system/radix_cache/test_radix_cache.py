import random


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("应抛 ValueError")


def test_insert_counts_new_tokens(m):
    c = m.PrefixCache()
    assert c.insert([1, 2, 3]) == 3
    assert c.insert([1, 2, 4]) == 1
    assert c.insert([1, 2]) == 0
    assert c.insert([]) == 0
    assert c.size() == 4


def test_match_vs_brute_force(m):
    rng = random.Random(0)
    c = m.PrefixCache()
    inserted = []
    for _ in range(60):
        s = [rng.randrange(4) for _ in range(rng.randint(1, 8))]
        c.insert(s)
        inserted.append(s)
    distinct = {tuple(s[:k]) for s in inserted for k in range(1, len(s) + 1)}
    assert c.size() == len(distinct)
    for _ in range(300):
        q = [rng.randrange(4) for _ in range(rng.randint(0, 10))]
        want = max((_lcp(q, s) for s in inserted), default=0)
        assert c.match_prefix(q) == want


def test_lru_eviction_order(m):
    c = m.PrefixCache()
    c.insert([1, 2, 3])   # t1
    c.insert([1, 4, 5])   # t2
    c.insert([6, 7])      # t3
    c.match_prefix([1, 2, 3])  # t4：刷新 1,2,3
    assert c.evict(1) == 1     # 叶子 3(t4) 5(t2) 7(t3) → 淘汰 5
    assert c.match_prefix([1, 4, 5]) == 2
    assert c.size() == 6
    # 上一行 match 刷新了 1、4（t5），此时叶子 3(t4) 4(t5) 7(t3) → 先淘汰 7，再淘汰 6
    assert c.evict(2) == 2
    assert c.match_prefix([6, 7]) == 0
    assert c.match_prefix([1, 2, 3]) == 3
    assert c.size() == 4


def test_locked_nodes_survive(m):
    c = m.PrefixCache()
    c.insert([1, 2, 3])
    c.insert([1, 2, 9])
    c.insert([5])
    c.lock([1, 2, 3])
    assert c.evict(100) == 2           # 只能淘汰 9 和 5
    assert c.size() == 3
    assert c.match_prefix([1, 2, 3]) == 3
    c.unlock([1, 2, 3])
    assert c.evict(100) == 3
    assert c.size() == 0


def test_lock_errors(m):
    c = m.PrefixCache()
    c.insert([1, 2])
    _expect_value_error(c.lock, [1, 2, 3])   # 没有完整缓存
    _expect_value_error(c.unlock, [1, 2])    # 没锁过
    c.lock([1, 2])
    c.lock([1])
    c.unlock([1, 2])
    assert c.evict(5) == 1                   # [1] 仍被锁，只能淘汰 2
    assert c.match_prefix([1, 2]) == 1


def test_evict_zero_and_partial(m):
    c = m.PrefixCache()
    c.insert([1, 2, 3, 4])
    assert c.evict(0) == 0
    assert c.evict(2) == 2          # 从叶子往上：4、3
    assert c.match_prefix([1, 2, 3, 4]) == 2
    assert c.size() == 2


def test_random_size_accounting(m):
    rng = random.Random(7)
    c = m.PrefixCache()
    size = 0
    locked = []
    for _ in range(400):
        r = rng.random()
        if r < 0.5:
            s = [rng.randrange(5) for _ in range(rng.randint(1, 6))]
            size += c.insert(s)
            if rng.random() < 0.2:
                c.lock(s)
                locked.append(s)
        elif r < 0.7 and locked:
            c.unlock(locked.pop(rng.randrange(len(locked))))
        else:
            k = rng.randint(0, 5)
            got = c.evict(k)
            assert 0 <= got <= k
            size -= got
        assert c.size() == size
        for s in locked:
            assert c.match_prefix(s) == len(s), "被锁定的序列不能被淘汰"
