class MedianFinder:
    def __init__(self):
        raise NotImplementedError

    def add(self, num) -> None:
        """加入一个数，O(log n)。"""
        raise NotImplementedError

    def median(self) -> float:
        """当前中位数，O(1)。偶数个取中间两数平均；为空时抛 ValueError。"""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
