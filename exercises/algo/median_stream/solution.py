import heapq


class MedianFinder:
    def __init__(self):
        self.lo = []  # 大顶堆（存相反数），较小的一半
        self.hi = []  # 小顶堆，较大的一半

    def add(self, num) -> None:
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self) -> float:
        if not self.lo:
            raise ValueError("median of empty stream")
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (-self.lo[0] + self.hi[0]) / 2

    def __len__(self) -> int:
        return len(self.lo) + len(self.hi)
