import random


class _Brute:
    """暴力版：淘汰时线性扫描 (freq, last_used) 最小的 key。"""

    def __init__(self, cap):
        self.cap, self.d, self.tick = cap, {}, 0

    def _use(self, k):
        self.tick += 1
        v, f, _ = self.d[k]
        self.d[k] = (v, f + 1, self.tick)

    def get(self, k):
        if k not in self.d:
            return -1
        self._use(k)
        return self.d[k][0]

    def put(self, k, v):
        if self.cap <= 0:
            return
        if k in self.d:
            _, f, t = self.d[k]
            self.d[k] = (v, f, t)
            self._use(k)
            return
        if len(self.d) >= self.cap:
            victim = min(self.d, key=lambda x: (self.d[x][1], self.d[x][2]))
            del self.d[victim]
        self.tick += 1
        self.d[k] = (v, 1, self.tick)


def test_leetcode_example(m):
    c = m.LFUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)
    assert c.get(2) == -1
    assert c.get(3) == 3
    c.put(4, 4)
    assert c.get(1) == -1
    assert c.get(3) == 3
    assert c.get(4) == 4
    assert len(c) == 2


def test_capacity_zero(m):
    c = m.LFUCache(0)
    c.put(1, 1)
    assert c.get(1) == -1
    assert len(c) == 0


def test_update_counts_as_use(m):
    c = m.LFUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.put(1, 10)  # key 1 次数变成 2
    c.put(3, 3)   # 淘汰次数为 1 的 key 2
    assert c.get(2) == -1
    assert c.get(1) == 10
    assert c.get(3) == 3


def test_tie_breaks_by_recency(m):
    c = m.LFUCache(3)
    for k in (1, 2, 3):
        c.put(k, k)
    c.get(2)
    c.get(1)
    c.get(3)      # 三个都是 2 次；最久未用的是 2
    c.put(4, 4)
    assert c.get(2) == -1
    assert c.get(1) == 1 and c.get(3) == 3 and c.get(4) == 4


def test_new_key_resets_min_freq(m):
    c = m.LFUCache(2)
    c.put(1, 1)
    for _ in range(5):
        c.get(1)
    c.put(2, 2)   # 次数 1
    c.put(3, 3)   # 淘汰 2（次数最少），而不是高频的 1
    assert c.get(2) == -1
    assert c.get(1) == 1
    assert c.get(3) == 3


def test_against_brute_force(m):
    rng = random.Random(53)
    for cap in (1, 2, 4, 7):
        c, b = m.LFUCache(cap), _Brute(cap)
        for _ in range(3000):
            k = rng.randrange(10)
            if rng.random() < 0.55:
                assert c.get(k) == b.get(k)
            else:
                v = rng.randrange(100)
                c.put(k, v)
                b.put(k, v)
            assert len(c) == len(b.d)
