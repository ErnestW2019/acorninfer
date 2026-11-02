"""PLD 式 n-gram 草稿器 + 贪心接受(spec §2.1)。纯 CPU,确定性,零 torch。

草稿器跑在引擎热路径:纯列表/dict 实现,无随机性来源——
索引值 = 后继 token 列表,严格按出现序 append(不 dedup,顺序即确定性来源);
`propose` 取后继列表前 K 位,不足补 `FILLER`。

确定性定案(报告口径):后继列表保持出现序、list 不 dedup(最简、可复现);
propose 输出 = 列表前 K + FILLER 补齐,同上下文同输出。
"""

FILLER = 0  # 固定填充 token(spec §2.1 定案)


class NgramProposer:
    """上下文 N-gram 索引:dict[n-gram tuple, 后继 list](按出现序,含重复)。"""

    def __init__(self, prompt_ids: list[int], n: int, k: int):
        self.n = n
        self.k = k
        self.context = list(prompt_ids)
        self.index: dict[tuple[int, ...], list[int]] = {}
        self._index_range(0, len(self.context))

    def _index_range(self, start: int, end: int) -> None:
        """位置 p 的 token 产生键 context[p-n:p](p >= n);增量 append O(1)/token。"""
        for p in range(max(start, self.n), end):
            key = tuple(self.context[p - self.n : p])
            self.index.setdefault(key, []).append(self.context[p])

    def extend(self, new_ids: list[int]) -> None:
        """生成 token 增量入索引(接受/续写后调用;键含新生成 token)。"""
        start = len(self.context)
        self.context.extend(new_ids)
        self._index_range(start, len(self.context))

    def propose(self) -> list[int]:
        """键 = 已提交上下文末 n 位(prompt+已生成);前 K 后继,不足补 FILLER。"""
        key = tuple(self.context[-self.n :])
        successors = self.index.get(key)
        out = list(successors[: self.k]) if successors else []
        out.extend([FILLER] * (self.k - len(out)))
        return out


def greedy_accept(next_token_ids: list[int], drafts: list[int]) -> tuple[int, int]:
    """最长前缀接受。入参 = argmax 后的 token ids(引擎侧先 logits.argmax)。

    返回 (a, bonus):a = drafts 与 next_token_ids 的最长前缀长度;
    bonus = next_token_ids[a](全中 a=K → 链尾第 K+1 位;零中 a=0 → 首位)。
    """
    a = 0
    for draft, actual in zip(drafts, next_token_ids):
        if draft != actual:
            break
        a += 1
    return a, next_token_ids[a]
