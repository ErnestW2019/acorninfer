"""调度器纯逻辑测试:准入顺序、水位、max_running、抢占最老(recompute)。

页面数学(易错,先算清楚):
- PAGE=4;mkseq(i, 5) → num_tokens=5 → 需 ceil(5/4)=2 块;
- 抢占场景 mkseq(i, 3, gen=1) → num_tokens=4 == 1 页满 → 下一 token 必须新块。
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


def test_fcfs_admission_order():
    alloc = BlockAllocator(8)
    sch = Scheduler(alloc, PAGE)
    for i in range(3):
        sch.add(mkseq(i, 5))
    prefills, decodes = sch.schedule()
    assert [s.seq_id for s in prefills] == [0, 1, 2]
    assert decodes == []


def test_max_running_cap():
    alloc = BlockAllocator(64)
    sch = Scheduler(alloc, PAGE, max_running=2)
    for i in range(3):
        sch.add(mkseq(i, 5))
    prefills, _ = sch.schedule()
    assert len(prefills) == 2
    assert sch.waiting[0].seq_id == 2


def test_admission_stops_at_watermark():
    alloc = BlockAllocator(8)
    sch = Scheduler(alloc, PAGE, max_running=4, watermark_blocks=3)
    for i in range(3):
        sch.add(mkseq(i, 5))  # 每条需 2 块
    prefills, _ = sch.schedule()
    # 准入数学:s0 后 free 8≥2 ✓;s1 后 8-4=4,预算 4-3=1 < 2 → s2 不准入
    assert [s.seq_id for s in prefills] == [0, 1]
    for s in sch.running:  # 模拟 prefill 已各占 2 块
        s.num_pages = 2
        s.block_table = [alloc.allocate(), alloc.allocate()]
    # free=4,水位 3 → 预算 1 < need 2 → 仍不准入
    prefills2, _ = sch.schedule()
    assert prefills2 == []
    sch.watermark = 0  # 水位放开 → 立即可入
    prefills3, _ = sch.schedule()
    assert [s.seq_id for s in prefills3] == [2]


def test_admission_reserves_decode_blocks():
    """页满 running 的本轮 decode 块必须被预留:准入不得吃掉它。

    repro:页满 s0(本轮 decode 要 +1 块)+ need=2 的 s1 → 无预留时会准入
    s1(2≥2),步内 s0 decode 无块可占 → BlockExhausted 崩溃。
    """
    alloc = BlockAllocator(3)
    sch = Scheduler(alloc, PAGE)
    s0 = mkseq(0, 3, gen=1)  # prefill 后 num_tokens=4 == 1 页满
    sch.add(s0)
    sch.schedule()  # s0 准入
    s0.status = RequestStatus.RUNNING
    s0.num_pages = 1
    s0.block_table = [alloc.allocate()]  # free=2
    s1 = mkseq(1, 8)  # need=2
    sch.add(s1)
    prefills, decodes = sch.schedule()
    # reserve=1(s0 页满):预算 2-0-1=1 < 2 → s1 留队,块留给 s0 本轮 decode
    assert prefills == []
    assert [s.seq_id for s in decodes] == [0]
    assert sch.waiting[0].seq_id == 1


def test_preempt_oldest_running_recompute():
    alloc = BlockAllocator(3)
    sch = Scheduler(alloc, PAGE)
    s0, s1 = mkseq(0, 3, gen=1), mkseq(1, 3, gen=1)  # num_tokens=4 == 1 页满
    for s in (s0, s1):
        sch.add(s)
    sch.schedule()  # 双双准入(此时尚未真实占块)
    for s in (s0, s1):
        s.status = RequestStatus.RUNNING
        s.num_pages = 1
        s.block_table = [alloc.allocate()]
    assert alloc.num_free == 1
    # 两条都页满:deficit=2 > free=1 → 抢占最老 s0(释放 1 块 → free=2 ≥ 1)
    prefills, decodes = sch.schedule()
    assert s0.status is RequestStatus.WAITING
    assert s0.seq_id == sch.waiting[0].seq_id  # 回到队首(FCFS 重试优先)
    assert s0.num_pages == 0 and s0.block_table == []
    assert [s.seq_id for s in decodes] == [1]
    assert prefills == []  # 刚被抢占者本轮不再准入(防抖;下轮重试)
    assert alloc.num_free == 2 and alloc.num_live == 1
