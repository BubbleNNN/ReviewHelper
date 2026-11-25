import heapq


def _count(nums):
    cnt = {}
    for x in nums:
        cnt[x] = cnt.get(x, 0) + 1
    return cnt


def top_k_heap(nums, k):
    if k <= 0:
        return []
    cnt = _count(nums)
    heap = []  # 最小堆，元素 (freq, -num)：堆顶是当前 Top-K 里“最差”的
    for num, f in cnt.items():
        item = (f, -num)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    heap.sort(reverse=True)  # 频次降序；同频次 -num 降序 = num 升序
    return [-neg for _, neg in heap]


def top_k_bucket(nums, k):
    if k <= 0:
        return []
    cnt = _count(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, f in cnt.items():
        buckets[f].append(num)
    out = []
    for f in range(len(nums), 0, -1):
        for num in sorted(buckets[f]):
            out.append(num)
            if len(out) == k:
                return out
    return out
