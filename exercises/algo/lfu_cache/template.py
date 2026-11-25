class LFUCache:
    def __init__(self, capacity: int):
        raise NotImplementedError

    def get(self, key: int) -> int:
        """命中返回 value 并使次数 +1；否则返回 -1。"""
        raise NotImplementedError

    def put(self, key: int, value: int) -> None:
        """写入或更新（更新也使次数 +1）。满了先淘汰次数最少、其中最久未用的 key。"""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
