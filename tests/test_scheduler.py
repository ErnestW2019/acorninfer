"""调度器纯逻辑测试:准入顺序、水位、max_running、抢占最老(recompute)。

页面数学(易错,先算清楚):
- PAGE=4;mkseq(i, 5) → num_tokens=5 → 需 ceil(5/4)=2 块;
- 抢占场景 mkseq(i, 3, gen=1):prefill 写 A=4 恰满 1 页,record 后
  num_tokens=5 > 1×4 → 本轮 decode 必须新块(判据是 >,见 scheduler.py)。
"""

from acorninfer.cache.pager import BlockAllocator
from acorninfer.engine.scheduler import Scheduler
from acorninfer.engine.sequence import RequestStatus, Sequence
from acorninfer.sampler import SamplingParams

PAGE = 4


def mkseq(sid, prompt_len, gen=0):
    s = Sequence(
        seq_id=sid,
        prompt_ids=[sid * 100 + i for i in range(prompt_len)],
        sampling=SamplingParams(),
        max_tokens=64,
        eos_ids=frozenset({999}),
    )
    s.generated_ids = list(range(1000, 1000 + gen))
    return s


def test_chunked_budget_must_be_positive():
    """chunked_budget ≤ 0 构造期 ValueError(0/-1 会让每步零预算空转)。"""
    for bad in (0, -1):
        try:
            Scheduler(BlockAllocator(8), PAGE, chunked_budget=bad)
        except ValueError as e:
            assert "chunked_budget" in str(e)
        else:
            raise AssertionError(f"chunked_budget={bad} 应构造期拒绝")


def test_fcfs_admission_order():
    alloc = BlockAllocator(8)
    sch = Scheduler(alloc, PAGE)
    for i in range(3):
        sch.add(mkseq(i, 5))
    chunks, decodes = sch.schedule()
    assert [(s.seq_id, start, ln) for s, start, ln in chunks] == [
        (0, 0, 5), (1, 0, 5), (2, 0, 5)
    ]
    assert decodes == []


def test_max_running_cap():
    alloc = BlockAllocator(64)
    sch = Scheduler(alloc, PAGE, max_running=2)
    for i in range(3):
        sch.add(mkseq(i, 5))
    chunks, _ = sch.schedule()
    assert len(chunks) == 2
    assert sch.waiting[0].seq_id == 2


def test_admission_stops_at_watermark():
    alloc = BlockAllocator(8)
    sch = Scheduler(alloc, PAGE, max_running=4, watermark_blocks=3)
    for i in range(3):
        sch.add(mkseq(i, 5))  # 每条需 2 块
    chunks, _ = sch.schedule()
    # 准入数学:s0 后 free 8≥2 ✓;s1 后 8-4=4,预算 4-3=1 < 2 → s2 不准入
    assert [s.seq_id for s, _, _ in chunks] == [0, 1]
    for s in sch.running:  # 模拟 prefill 已各占 2 块(完成态:有生成 token)
        s.num_pages = 2
        s.block_table = [alloc.allocate(), alloc.allocate()]
        s.generated_ids.append(9999)
    # free=4,水位 3 → 预算 1 < need 2 → 仍不准入
    chunks2, _ = sch.schedule()
    assert chunks2 == []
    sch.watermark = 0  # 水位放开 → 立即可入
    chunks3, _ = sch.schedule()
    assert [(s.seq_id, start, ln) for s, start, ln in chunks3] == [(2, 0, 5)]


def test_admission_reserves_decode_blocks():
    """页满 running 的本轮 decode 块必须被预留:准入不得吃掉它。

    repro:页满 s0(本轮 decode 要 +1 块)+ need=2 的 s1 → 无预留时会准入
    s1(2≥2),步内 s0 decode 无块可占 → BlockExhausted 崩溃。

    状态语义:prefill 写入 A=4(prompt 3 + gen 1)恰好满 1 页,record 后
    num_tokens=5 > 1×4 → 本轮 decode 真要块。
    """
    alloc = BlockAllocator(3)
    sch = Scheduler(alloc, PAGE)
    s0 = mkseq(0, 3, gen=1)
    sch.add(s0)
    sch.schedule()  # s0 准入
    s0.status = RequestStatus.RUNNING
    s0.num_pages = 1
    s0.block_table = [alloc.allocate()]  # free=2;KV 已写 A=4,满页
    s0.generated_ids.append(9999)  # prefill 选出新 token → num_tokens=5
    s1 = mkseq(1, 8)  # need=2
    sch.add(s1)
    prefills, decodes = sch.schedule()
    # reserve=1(s0 本轮 decode 要块):预算 2-0-1=1 < 2 → s1 留队
    assert prefills == []
    assert [s.seq_id for s in decodes] == [0]
    assert sch.waiting[0].seq_id == 1


def test_preempt_oldest_running_recompute():
    alloc = BlockAllocator(3)
    sch = Scheduler(alloc, PAGE)
    s0, s1 = mkseq(0, 3, gen=1), mkseq(1, 3, gen=1)  # prefill 写 A=4,恰满 1 页
    for s in (s0, s1):
        sch.add(s)
    sch.schedule()  # 双双准入(此时尚未真实占块)
    for s in (s0, s1):
        s.status = RequestStatus.RUNNING
        s.num_pages = 1
        s.block_table = [alloc.allocate()]
        s.generated_ids.append(9999)  # record 后 num_tokens=5 > 1×4:本轮 decode 要新块
    assert alloc.num_free == 1
    # 两条都溢出页容量:deficit=2 > free=1 → 抢占最老 s0(释放 1 块 → free=2 ≥ 1)
    prefills, decodes = sch.schedule()
    assert s0.status is RequestStatus.WAITING
    assert s0.seq_id == sch.waiting[0].seq_id  # 回到队首(FCFS 重试优先)
    assert s0.num_pages == 0 and s0.block_table == []
    assert [s.seq_id for s in decodes] == [1]
    assert prefills == []  # 刚被抢占者本轮不再准入(防抖;下轮重试)
    assert alloc.num_free == 2 and alloc.num_live == 1


# ---- characterization(M5 Task 1):钉住 chunk 化之前的 schedule() 现状语义 ----
# 字面断言:准入顺序 / 全量 prompt / (prefills, decodes) 配对结构。scheduler
# 契约升级为 (chunks, decodes)(chunks=[(seq, start, length)])后,这些测试
# 机械适配解包形式、语义原样 —— 它们就是 budget=None 等价性的证明钉子。


def test_char_mixed_running_and_waiting():
    """1 条 decode 级 running + 2 条异长 waiting:准入按 FCFS,全量 prompt。"""
    alloc = BlockAllocator(16)
    sch = Scheduler(alloc, PAGE)
    s0 = mkseq(0, 3, gen=1)
    sch.add(s0)
    sch.schedule()  # s0 准入
    s0.status = RequestStatus.RUNNING
    s0.num_pages = 1
    s0.block_table = [alloc.allocate()]
    s0.generated_ids.append(9999)  # 进入 decode 级
    s1, s2 = mkseq(1, 5), mkseq(2, 8)
    sch.add(s1)
    sch.add(s2)
    chunks, decodes = sch.schedule()
    # FCFS 准入 + 全量 prompt 一步一条 chunk(start=0,budget=None 等价钉子)
    assert [(s.seq_id, start, ln) for s, start, ln in chunks] == [(1, 0, 5), (2, 0, 8)]
    assert [s.seq_id for s in decodes] == [0]
    assert all(c[0] is s for c, s in zip(chunks, (s1, s2)))  # 同一对象


def test_char_single_full_prompt_admitted():
    alloc = BlockAllocator(8)
    sch = Scheduler(alloc, PAGE)
    s0 = mkseq(0, 9)  # 跨 3 页,单步全量准入
    sch.add(s0)
    chunks, decodes = sch.schedule()
    assert [(s.seq_id, start, ln) for s, start, ln in chunks] == [(0, 0, 9)]
    assert decodes == []
    assert chunks[0][0] is s0  # 全量,一步准入


def test_char_preempt_mid_prefill_free_and_readmit_front():
    """抢占边界(与 mid-prefill 语义交接):被抢者释放全部块、回队首重算。"""
    alloc = BlockAllocator(4)
    sch = Scheduler(alloc, PAGE)
    s0 = mkseq(0, 12)
    sch.add(s0)
    sch.schedule()  # s0 准入(need=3)
    s0.status = RequestStatus.RUNNING
    s0.num_pages = 3
    s0.block_table = [alloc.allocate() for _ in range(3)]  # free=1;KV 写满 12 token
    s0.generated_ids.append(8888)  # num_tokens=13 > 3×4 → 本轮 decode 缺 1 块
    stolen = alloc.allocate()  # 外部占 1 页 → free=0 < deficit=1 → 抢占 s0
    chunks, decodes = sch.schedule()
    assert s0.status is RequestStatus.WAITING
    assert sch.waiting[0].seq_id == 0  # 回队首(FCFS 重算优先)
    assert s0.num_pages == 0 and s0.block_table == []  # 块全释放
    assert s0.prefill_cursor == 0  # 重算从头(M5:cursor 随 _release 归零)
    assert chunks == [] and decodes == []  # 刚被抢者本轮不准入(防抖)
    alloc.free(stolen)
    chunks2, _ = sch.schedule()  # 下轮队首重准入(全量重算:含生成 token)
    assert [(s.seq_id, start, ln) for s, start, ln in chunks2] == [(0, 0, 13)]


# ---- M5 Task 1:chunked prefill 契约(chunks = [(seq, start, length)])----
# 先红后绿:本组测试钉预算 FCFS 切分 / 页账保护 / 抢占归零 / None 等价。


def mk_mid(seq, cursor, alloc, pages=None):
    """模拟引擎在 chunk 写入后的 mid-prefill 状态(cursor/pages/块表)。"""
    seq.prefill_cursor = cursor
    seq.num_pages = pages if pages is not None else -(-cursor // PAGE)
    seq.block_table = [alloc.allocate() for _ in range(seq.num_pages)]
    seq.status = RequestStatus.RUNNING


def test_chunks_budget_none_full_prompt():
    """budget=None(默认):每序列一条全量 chunk(start=0),等价 chunk 化之前。"""
    alloc = BlockAllocator(16)
    sch = Scheduler(alloc, PAGE)
    for i, n in enumerate((5, 8)):
        sch.add(mkseq(i, n))
    chunks, decodes = sch.schedule()
    assert [(s.seq_id, start, ln) for s, start, ln in chunks] == [(0, 0, 5), (1, 0, 8)]
    assert decodes == []


def test_chunks_budget_running_midprefill_first():
    """FCFS:running mid-prefill 先吃预算,waiting 新准入拿剩余。"""
    alloc = BlockAllocator(64)
    sch = Scheduler(alloc, PAGE, chunked_budget=4)
    s0 = mkseq(0, 9)
    sch.add(s0)
    chunks, _ = sch.schedule()
    assert [(s.seq_id, start, ln) for s, start, ln in chunks] == [(0, 0, 4)]
    mk_mid(s0, 4, alloc)  # 引擎写入 4 token 后的页态(1 页)
    s1 = mkseq(1, 6)
    sch.add(s1)
    chunks2, decodes2 = sch.schedule()
    # s0 续读 (4,4) 吃满预算 → s1 本步不准入(队头阻塞,FCFS 保序)
    assert [(s.seq_id, start, ln) for s, start, ln in chunks2] == [(0, 4, 4)]
    assert decodes2 == [] and sch.waiting[0].seq_id == 1
    mk_mid(s0, 8, alloc)
    chunks3, _ = sch.schedule()  # s0 尾块 (8,1) 吃 1,余 3 → s1 首块 (0,3)
    assert [(s.seq_id, start, ln) for s, start, ln in chunks3] == [(0, 8, 1), (1, 0, 3)]
    mk_mid(s0, 9, alloc, pages=3)
    s0.generated_ids.append(7001)  # prefill 完成 → decode 级
    mk_mid(s1, 3, alloc)  # s1 首块已落地(cursor=3,1 页)
    chunks4, _ = sch.schedule()
    assert [(s.seq_id, start, ln) for s, start, ln in chunks4] == [(1, 3, 3)]


def test_chunks_budget_waiting_partial_first_chunk():
    """waiting 首块可被预算截断(budget < prompt 长度)。"""
    alloc = BlockAllocator(64)
    sch = Scheduler(alloc, PAGE, chunked_budget=3)
    sch.add(mkseq(0, 9))
    chunks, _ = sch.schedule()
    assert [(s.seq_id, start, ln) for s, start, ln in chunks] == [(0, 0, 3)]


def test_chunks_one_per_seq_per_step():
    """每序列每步至多一个 chunk;budget=16:9+7 两序列吃满,第三个不准入。"""
    alloc = BlockAllocator(64)
    sch = Scheduler(alloc, PAGE, chunked_budget=16)
    for i in range(3):
        sch.add(mkseq(i, 9))
    chunks, _ = sch.schedule()
    assert [(s.seq_id, start, ln) for s, start, ln in chunks] == [(0, 0, 9), (1, 0, 7)]
    assert sch.waiting[0].seq_id == 2  # 预算耗尽,FCFS 队头阻塞


def test_chunks_midprefill_pages_reserved_not_starved():
    """mid-prefill 续读步的追加页入 reserve:新准入不得吃掉(否则步内
    BlockExhausted —— chunk 化之前不存在的崩法);预算 FCFS 不饿死在跑者。"""
    alloc = BlockAllocator(4)
    sch = Scheduler(alloc, PAGE, chunked_budget=8)
    s0 = mkseq(0, 9)
    sch.add(s0)
    chunks, _ = sch.schedule()  # (0,8);need=3 ≤ free 4
    assert [(s.seq_id, start, ln) for s, start, ln in chunks] == [(0, 0, 8)]
    mk_mid(s0, 8, alloc)  # 2 页,free=2
    s1 = mkseq(1, 8)  # need=2
    sch.add(s1)
    chunks2, _ = sch.schedule()
    # s0 尾块 (8,1) 追加 1 页 → reserve=1:free 2-1=1 < 2 → s1 不准入
    # (无此保护 s1 会准入,步内 s0 追加 + s1 写 2 页 > free 2 → BlockExhausted)
    assert [(s.seq_id, start, ln) for s, start, ln in chunks2] == [(0, 8, 1)]
    assert sch.waiting[0].seq_id == 1
    # 尾块落地:引擎追加第 3 页(free 2→1),cursor=9,prompt 9 token 全程完成
    s0.block_table.append(alloc.allocate())
    s0.num_pages = 3
    s0.prefill_cursor = 9
    s0.generated_ids.append(7001)
    sch.schedule()  # s0 进 decode 级;页随完成释放后 s1 照常准入(后续步)


def test_chunks_preempt_resets_cursor():
    """抢占 mid-prefill 受害者:_release 处 prefill_cursor 归零(重算从头)。"""
    alloc = BlockAllocator(4)
    sch = Scheduler(alloc, PAGE, chunked_budget=8)
    s0 = mkseq(0, 9)
    sch.add(s0)
    sch.schedule()
    mk_mid(s0, 8, alloc)  # 2 页,free=2
    s1 = mkseq(1, 5, gen=1)  # decode 级页满:本轮 decode 要 +1 块
    sch.running.append(s1)
    s1.status = RequestStatus.RUNNING
    s1.num_pages = 1
    s1.block_table = [alloc.allocate()]  # free=1
    s1.generated_ids.append(9999)  # num_tokens=6 > 1×4 → deficit=1
    # 需求 = deficit 1 + s0 尾块追加 1 = 2 > free 1 → 抢最老 s0
    chunks, decodes = sch.schedule()
    assert s0.prefill_cursor == 0 and s0.num_pages == 0 and s0.block_table == []
    assert s0.status is RequestStatus.WAITING and sch.waiting[0].seq_id == 0
    assert chunks == [] and [s.seq_id for s in decodes] == [1]


def test_chunks_midprefill_chunked_not_in_decodes():
    """chunked 的 mid-prefill 序列不入本步 decodes(decodes 只收 decode 级);
    预算同时挡住新准入,s1 留 waiting(FCFS 队头阻塞)。"""
    alloc = BlockAllocator(64)
    sch = Scheduler(alloc, PAGE, chunked_budget=4)
    s0, s1 = mkseq(0, 9), mkseq(1, 9)
    sch.add(s0)
    sch.add(s1)
    sch.schedule()  # s0 (0,4) 吃满预算;s1 不准入
    mk_mid(s0, 4, alloc)
    chunks, decodes = sch.schedule()
    assert [s.seq_id for s, _, _ in chunks] == [0]
    assert decodes == []  # s0 在 chunk 名单;s1 留 waiting
