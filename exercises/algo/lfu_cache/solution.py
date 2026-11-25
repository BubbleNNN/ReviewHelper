from collections import OrderedDict, defaultdict


class LFUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.kv = {}  # key -> [value, freq]
        self.buckets = defaultdict(OrderedDict)  # freq -> keys，头部最久未用
        self.min_freq = 0

    def _touch(self, key):
        val, f = self.kv[key]
        bucket = self.buckets[f]
        del bucket[key]
        if not bucket:
            del self.buckets[f]
            if self.min_freq == f:
                self.min_freq = f + 1
        self.kv[key][1] = f + 1
        self.buckets[f + 1][key] = None

    def get(self, key: int) -> int:
        if key not in self.kv:
            return -1
        self._touch(key)
        return self.kv[key][0]

    def put(self, key: int, value: int) -> None:
        if self.cap <= 0:
            return
        if key in self.kv:
            self.kv[key][0] = value
            self._touch(key)
            return
        if len(self.kv) >= self.cap:
            bucket = self.buckets[self.min_freq]
            old, _ = bucket.popitem(last=False)
            if not bucket:
                del self.buckets[self.min_freq]
            del self.kv[old]
        self.kv[key] = [value, 1]
        self.buckets[1][key] = None
        self.min_freq = 1

    def __len__(self) -> int:
        return len(self.kv)
