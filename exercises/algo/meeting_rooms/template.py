def min_meeting_rooms(intervals):
    """半开区间 [start, end)，返回最少需要的房间数。"""
    raise NotImplementedError


def assign_rooms(intervals):
    """按题面规则给每个会议分配房间号，返回值与输入顺序一一对应。

    规则：按 (start, 输入下标) 处理；结束时间 <= start 的房间空闲；
    优先用编号最小的空闲房间，否则开新房（编号 = 已开房间数）。
    """
    raise NotImplementedError
