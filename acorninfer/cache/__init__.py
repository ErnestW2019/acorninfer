"""KV cache package(M2):M1 朴素 KVCache + M2 分页 pager。

`acorninfer.cache.KVCache` 兼容导入保持不变(M1 的 generate.py 与等价性测试仍用它);
分页分配器/页式缓存在 `acorninfer.cache.pager`。
"""

from acorninfer.cache.kv import KVCache

__all__ = ["KVCache"]
