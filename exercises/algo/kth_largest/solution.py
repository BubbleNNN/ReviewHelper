import heapq
import random


def kth_largest_quickselect(nums, k):
    arr = list(nums)
    t = len(arr) - k  # 升序下标
    rng = random.Random(0)
    while True:
        pivot = arr[rng.randrange(len(arr))]
        lt = [x for x in arr if x < pivot]
        eq_n = sum(1 for x in arr if x == pivot)
        if t < len(lt):
            arr = lt
        elif t < len(lt) + eq_n:
            return pivot
        else:
            t -= len(lt) + eq_n
            arr = [x for x in arr if x > pivot]


def kth_largest_heap(nums, k):
    heap = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    return heap[0]
