import heapq
import itertools


class _Node:
    __slots__ = ("children", "parent", "key", "ref", "last")

    def __init__(self, parent=None, key=None):
        self.children = {}
        self.parent, self.key = parent, key
        self.ref = 0
        self.last = 0


class PrefixCache:
    """按 token 建的前缀树（每个节点一个 token）。"""

    def __init__(self):
        self.root = _Node()
        self.clock = 0
        self._size = 0
        self._tie = itertools.count()

    def _tick(self):
        self.clock += 1
        return self.clock

    def _walk(self, tokens):
        """返回沿 tokens 能走到的节点路径（不含 root）。"""
        node, path = self.root, []
        for t in tokens:
            node = node.children.get(t)
            if node is None:
                break
            path.append(node)
        return path

    def insert(self, tokens):
        now = self._tick()
        node, added = self.root, 0
        for t in tokens:
            child = node.children.get(t)
            if child is None:
                child = _Node(node, t)
                node.children[t] = child
                added += 1
            child.last = now
            node = child
        self._size += added
        return added

    def match_prefix(self, tokens):
        now = self._tick()
        path = self._walk(tokens)
        for n in path:
            n.last = now
        return len(path)

    def lock(self, tokens):
        path = self._walk(tokens)
        if len(path) != len(tokens):
            raise ValueError("只能锁定已完整缓存的序列")
        now = self._tick()
        for n in path:
            n.ref += 1
            n.last = now

    def unlock(self, tokens):
        path = self._walk(tokens)
        if len(path) != len(tokens) or any(n.ref == 0 for n in path):
            raise ValueError("unlock 的序列没有被锁定")
        for n in path:
            n.ref -= 1

    def evict(self, n_tokens):
        heap = []

        def push(node):
            heapq.heappush(heap, (node.last, next(self._tie), node))

        stack = [self.root]
        while stack:  # 收集所有可淘汰的叶子
            node = stack.pop()
            for c in node.children.values():
                if c.children:
                    stack.append(c)
                elif c.ref == 0:
                    push(c)
        evicted = 0
        while evicted < n_tokens and heap:
            _, _, node = heapq.heappop(heap)
            parent = node.parent
            del parent.children[node.key]
            evicted += 1
            if parent is not self.root and not parent.children and parent.ref == 0:
                push(parent)
        self._size -= evicted
        return evicted

    def size(self):
        return self._size
