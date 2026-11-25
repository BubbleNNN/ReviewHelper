import random


def _max_overlap(intervals):
    """暴力：最大重叠一定发生在某个会议的开始时刻。"""
    best = 0
    for t, _ in intervals:
        best = max(best, sum(1 for s, e in intervals if s <= t < e))
    return best


def _simulate(intervals):
    """O(n * rooms) 的直接模拟：每个房间记录它最后一个会议的结束时间。"""
    room_end = []
    out = [None] * len(intervals)
    for i in sorted(range(len(intervals)), key=lambda i: (intervals[i][0], i)):
        s, e = intervals[i]
        free = [r for r, end in enumerate(room_end) if end <= s]
        if free:
            r = min(free)
            room_end[r] = e
        else:
            r = len(room_end)
            room_end.append(e)
        out[i] = r
    return out


def _no_conflict(intervals, rooms):
    by_room = {}
    for (s, e), r in zip(intervals, rooms):
        by_room.setdefault(r, []).append((s, e))
    for lst in by_room.values():
        lst.sort()
        for (s1, e1), (s2, _) in zip(lst, lst[1:]):
            if s2 < e1:
                return False
    return True


def test_examples(m):
    ivs = [[0, 30], [5, 10], [15, 20]]
    assert m.min_meeting_rooms(ivs) == 2
    assert m.assign_rooms(ivs) == [0, 1, 1]
    assert m.min_meeting_rooms([[7, 10], [2, 4]]) == 1


def test_half_open_boundary(m):
    ivs = [[1, 5], [5, 10], [10, 12]]
    assert m.min_meeting_rooms(ivs) == 1
    assert m.assign_rooms(ivs) == [0, 0, 0]


def test_empty(m):
    assert m.min_meeting_rooms([]) == 0
    assert m.assign_rooms([]) == []


def test_smallest_free_room_is_reused(m):
    # 0,1,2 号房同时占用；1 号和 2 号先后空出，新会议要拿 1 号
    ivs = [[0, 10], [0, 3], [0, 4], [5, 8], [6, 9]]
    assert m.assign_rooms(ivs) == [0, 1, 2, 1, 2]


def test_same_start_tie_by_index(m):
    ivs = [[0, 5], [0, 2], [0, 9], [2, 4]]
    assert m.assign_rooms(ivs) == [0, 1, 2, 1]


def test_random_against_brute_force(m):
    rng = random.Random(41)
    for _ in range(400):
        ivs = []
        for _ in range(rng.randint(1, 15)):
            s = rng.randint(0, 20)
            ivs.append([s, s + rng.randint(1, 8)])
        k = _max_overlap(ivs)
        assert m.min_meeting_rooms(ivs) == k
        rooms = m.assign_rooms(ivs)
        assert rooms == _simulate(ivs)
        assert _no_conflict(ivs, rooms)
        assert len(set(rooms)) == k
