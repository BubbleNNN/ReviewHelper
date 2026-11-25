import random


def test_example(m):
    c = m.LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)
    assert c.get(2) == -1
    c.put(4, 4)
    assert c.get(1) == -1
    assert c.get(3) == 3
    assert c.get(4) == 4
    assert len(c) == 2


def test_update_refreshes_recency(m):
    c = m.LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.put(1, 10)  # 更新也算使用
    c.put(3, 3)   # 淘汰 2
    assert c.get(1) == 10
    assert c.get(2) == -1
    assert len(c) == 2


def test_capacity_one(m):
    c = m.LRUCache(1)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == -1
    assert c.get(2) == 2


def test_against_brute_force(m):
    rng = random.Random(42)
    for cap in (1, 3, 8):
        c = m.LRUCache(cap)
        order = []  # 暴力：list 末尾 = 最近使用
        store = {}
        for _ in range(3000):
            k = rng.randrange(12)
            if rng.random() < 0.5:
                got = c.get(k)
                want = store.get(k, -1)
                if k in store:
                    order.remove(k)
                    order.append(k)
                assert got == want
            else:
                v = rng.randrange(1000)
                c.put(k, v)
                if k in store:
                    order.remove(k)
                elif len(store) >= cap:
                    old = order.pop(0)
                    del store[old]
                store[k] = v
                order.append(k)
            assert len(c) == len(store)


def test_no_ordered_dict(m):
    import inspect
    src = inspect.getsource(m)
    assert "OrderedDict" not in src, "这题要求自己实现双向链表"
