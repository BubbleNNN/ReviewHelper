class PrefixCache:
    """token 级前缀缓存（trie / radix tree），支持引用计数和 LRU 淘汰。

    逻辑时钟：每次 insert / match_prefix / lock 调用时钟 +1，
    并把这次访问路径上所有节点的“最近访问时间”设为当前时钟。
    """

    def __init__(self):
        raise NotImplementedError

    def insert(self, tokens):
        """插入一条 token 序列，返回新增的 token（节点）数。"""
        raise NotImplementedError

    def match_prefix(self, tokens):
        """返回已缓存的最长前缀长度，并刷新命中路径的访问时间。"""
        raise NotImplementedError

    def lock(self, tokens):
        """路径上每个节点引用计数 +1（正在被请求使用，不能淘汰）。未完整缓存时 ValueError。"""
        raise NotImplementedError

    def unlock(self, tokens):
        """路径上每个节点引用计数 -1。没有被锁定时 ValueError。"""
        raise NotImplementedError

    def evict(self, n_tokens):
        """按 LRU 淘汰引用计数为 0 的叶子，直到淘汰 >= n_tokens 个或无可淘汰。返回淘汰数。"""
        raise NotImplementedError

    def size(self):
        """当前缓存的 token（节点）总数。"""
        raise NotImplementedError
