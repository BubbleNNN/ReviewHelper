import heapq


def min_meeting_rooms(intervals):
    ends = []
    best = 0
    for s, e in sorted(intervals):
        while ends and ends[0] <= s:
            heapq.heappop(ends)
        heapq.heappush(ends, e)
        best = max(best, len(ends))
    return best


def assign_rooms(intervals):
    order = sorted(range(len(intervals)), key=lambda i: (intervals[i][0], i))
    busy = []  # (end, room)
    free = []  # 空闲房间号，最小堆
    opened = 0
    out = [None] * len(intervals)
    for i in order:
        s, e = intervals[i]
        while busy and busy[0][0] <= s:
            _, room = heapq.heappop(busy)
            heapq.heappush(free, room)
        if free:
            room = heapq.heappop(free)
        else:
            room = opened
            opened += 1
        out[i] = room
        heapq.heappush(busy, (e, room))
    return out
