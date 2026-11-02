"""M8 Task 0:n-gram 草稿器 + 贪心接受(纯 CPU,零 torch;spec §2.1)。"""

from acorninfer.spec.ngram import FILLER, NgramProposer, greedy_accept


def test_index_build_and_lookup():
    """prompt 建索引:键 = N-gram tuple,值 = 后继(按出现序,重复上下文先后都记录);查表命中/未命中。"""
    p = NgramProposer([1, 2, 3, 1, 2, 4], n=2, k=2)
    # 键 = 2-gram tuple,值 = 后继列表,按出现序:(1,2) 出现两次 → [3, 4]
    assert p.index[(1, 2)] == [3, 4]
    assert p.index[(2, 3)] == [1]
    assert p.index[(3, 1)] == [2]
    assert (4, 1) not in p.index  # 末尾 n-gram 无后继,不入索引

    # 命中:上下文末 n 位 = (1,2) → 后继按出现序取前 K
    hit = NgramProposer([7, 8, 9, 7, 8], n=2, k=3)
    assert hit.propose() == [9, FILLER, FILLER]
    # 未命中:末 n 位无后继记录 → 全 FILLER
    assert NgramProposer([5, 6, 7], n=2, k=2).propose() == [FILLER, FILLER]


def test_propose_deterministic_and_padded():
    """同上下文同输出;候选 <K 用 filler=0 补齐(spec §2.1);K 候选截断。"""
    p = NgramProposer([1, 2, 1, 2, 1, 2], n=2, k=3)
    # (1,2) 后继 = [1, 1](按出现序)→ 不足 K 补 FILLER
    assert p.propose() == [1, 1, FILLER]
    # 确定性:同上下文多次 propose 同输出,且无随机性来源
    assert p.propose() == p.propose() == p.propose()

    # 候选 > K 截断到 K:(1,2) 后继 = [3, 4, 5](上下文末 n 位 = (1,2)),K=2 → 只取前两个
    q = NgramProposer([1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2], n=2, k=2)
    assert q.propose() == [3, 4]


def test_propose_uses_prompt_plus_generated():
    """索引随生成增量更新:生成 token 进入上下文后可被查到(增量 append O(1))。"""
    # prompt 末尾 (3,4) 无后继 → 全 FILLER
    p = NgramProposer([1, 2, 3, 4], n=2, k=3)
    assert p.propose() == [FILLER, FILLER, FILLER]

    # 生成 [2,3,4] 入上下文后:(3,4)→2 成为新键;上下文末 n 位恰 = (3,4) → 可查到
    p.extend([2, 3, 4])
    assert p.index[(3, 4)] == [2]      # 增量入索引
    assert p.index[(2, 3)] == [4, 4]   # prompt 旧键 + 生成新后继,出现序追加
    assert p.propose() == [2, FILLER, FILLER]


def test_greedy_accept_longest_prefix():
    """greedy_accept(argmax_ids[m], drafts[K]) -> (a, bonus):最长前缀;全中给 bonus;零中 a=0 仍产 bonus。"""
    # 部分中:最长前缀 a=2,bonus = argmax_ids[a](首个不匹配位)
    a, bonus = greedy_accept([5, 6, 9, 42], [5, 6, 7])
    assert (a, bonus) == (2, 9)
    # 全中:a=K,bonus = next_token_ids[K](第 K+1 位,链尾)
    a, bonus = greedy_accept([5, 6, 7, 42], [5, 6, 7])
    assert (a, bonus) == (3, 42)
    # 零中:a=0,仍产 bonus = next_token_ids[0]
    a, bonus = greedy_accept([9, 8, 7], [1, 2])
    assert (a, bonus) == (0, 9)


def test_accept_is_exactly_greedy_chain():
    """黄金性质钉:构造 logits 使贪心链逐 token == drafts → (a=K, bonus=链尾);
    任何 draft 与 argmax 不符即断,其后位置即使巧合相符也不计(前缀语义)。"""
    vocab = 10

    def argmax_ids(rows):
        # 引擎侧流程镜像:logits 行 → argmax → token ids(纯 CPU 可测)
        return [max(range(vocab), key=lambda v: row[v]) for row in rows]

    # 链逐位 == drafts:logits 行 argmax = [3,1,4,1,链尾 5] → (K=4, bonus=5)
    rows = [[1.0 if v == t else 0.0 for v in range(vocab)]
            for t in (3, 1, 4, 1, 5)]
    assert greedy_accept(argmax_ids(rows), [3, 1, 4, 1]) == (4, 5)

    # 第 3 位(0 基 2)断链:其后第 4 位巧合 == drafts[3] 也不计 → 前缀语义(非集合匹配)
    rows = [[1.0 if v == t else 0.0 for v in range(vocab)]
            for t in (3, 1, 9, 1, 5)]
    assert greedy_accept(argmax_ids(rows), [3, 1, 4, 1]) == (2, 9)
