from collections import deque


def max_sliding_window(nums, k):
    dq = deque()  # 存下标，对应值从队头到队尾递减
    out = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
