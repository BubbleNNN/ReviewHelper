class _Node:
    __slots__ = ("key", "val", "prev", "next")

    def __init__(self, key=None, val=None):
        self.key, self.val = key, val
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}
        self.head, self.tail = _Node(), _Node()  # 哨兵：head.next 最新，tail.prev 最旧
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _add_front(self, node):
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        node = self.map.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._add_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        node = self.map.get(key)
        if node is not None:
            node.val = value
            self._remove(node)
            self._add_front(node)
            return
        if self.cap <= 0:
            return
        if len(self.map) >= self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]
        node = _Node(key, value)
        self.map[key] = node
        self._add_front(node)

    def __len__(self) -> int:
        return len(self.map)
