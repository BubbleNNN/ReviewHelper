class Trie:
    def __init__(self):
        raise NotImplementedError

    def insert(self, word: str) -> None:
        """插入一个词；重复插入只算一次。"""
        raise NotImplementedError

    def search(self, word: str) -> bool:
        """word 是否被完整插入过。"""
        raise NotImplementedError

    def starts_with(self, prefix: str) -> bool:
        """是否存在以 prefix 开头的已插入词。"""
        raise NotImplementedError

    def longest_prefix(self, word: str) -> int:
        """沿 word 的字符从根往下最多能走几步。"""
        raise NotImplementedError

    def complete(self, prefix: str, limit=None) -> list:
        """以 prefix 开头的已插入词，字典序；limit 不为 None 时最多 limit 个。"""
        raise NotImplementedError
