def _need(nums, cap):
    """贪心：每段和不超过 cap 时最少需要几段。"""
    segs, cur = 1, 0
    for x in nums:
        if cur + x > cap:
            segs += 1
            cur = x
        else:
            cur += x
    return segs


def split_array(nums, k):
    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if _need(nums, mid) <= k:
            hi = mid
        else:
            lo = mid + 1
    return lo


def partition(nums, k):
    cap = split_array(nums, k)
    n = len(nums)
    segs, cur, cur_sum = [], [], 0
    for i, x in enumerate(nums):
        segs_left = k - len(segs)  # 包括当前正在填的这一段
        must_cut = (n - i) <= segs_left - 1  # 剩下的元素刚好够后面每段一个
        if cur and (cur_sum + x > cap or must_cut):
            segs.append(cur)
            cur, cur_sum = [], 0
        cur.append(x)
        cur_sum += x
    segs.append(cur)
    return segs
