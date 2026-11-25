class _Node:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = _Node()
        self.size = 0

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, _Node())
        if not node.is_end:
            node.is_end = True
            self.size += 1

    def _walk(self, s):
        node = self.root
        for ch in s:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        if prefix == "":
            return self.size > 0
        return self._walk(prefix) is not None

    def longest_prefix(self, word: str) -> int:
        node, n = self.root, 0
        for ch in word:
            node = node.children.get(ch)
            if node is None:
                break
            n += 1
        return n

    def complete(self, prefix: str, limit=None) -> list:
        out = []
        if limit is not None and limit <= 0:
            return out
        start = self._walk(prefix)
        if start is None:
            return out
        # 显式栈做先序 DFS；孩子逆序压栈，保证按字典序弹出
        stack = [(start, prefix)]
        while stack:
            node, s = stack.pop()
            if node.is_end:
                out.append(s)
                if limit is not None and len(out) >= limit:
                    break
            for ch in sorted(node.children, reverse=True):
                stack.append((node.children[ch], s + ch))
        return out
