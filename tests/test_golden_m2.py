"""M2 acceptance(spec §4/§5):并发引擎输出正确 + KV 无泄漏。

- 并发 greedy 与逐请求参考(generate_cached,M1 路径)逐 id 一致:一次性提交
  全部 prompt,小池子(512 块 / max_running 4 / watermark 2)强制批处理并
  有可能触发抢占 —— 抢占 recompute 必须保真,输出不得因此改变;比较按
  seq_id 对号,与完成顺序无关;
- 混合负载 fuzz:20 条随机长度请求分 4 波 mid-flight 插入,全部完成后
  stop_reason 合法、allocator 零 live 块、drop 全部后 engine.seqs 为空。

prompt 选择沿用 M0 先例:事实续写型、经实测无 bf16 argmax 平局。这里平局
是真的普遍:24 条候选 prompt 各跑 16 步贪心,逐步 top-2 margin 全域
0.0-0.5(bf16 ulp @|logit|≈25 就是 0.125),中文候选 ≤0.125 尤其密集。
已换掉的实测平局:「太阳系中最大的行星是木星…」并发(batch-4 分页 decode)
在第 3 个生成步选 99558、顺序参考选 108042 —— teacher-forced 下两者恰为
top-2,margin 0.125(1 ulp),属跨路径量化翻转,非引擎 bug(该步显存充裕,
不可能触发抢占);「地球绕太阳…」margin 0.0 同理弃用。入选的 4 条是池内
headroom 最大的实测组合。若未来失败且首个分歧步的真值 top-2 margin <
~0.25,仍是真值平局,换 prompt 并在此记录;系统性错位 = 引擎 bug。

单卡顺序协议:GPU 0 被外部服务占用,统一 CUDA_VISIBLE_DEVICES=1;
module 级 fixture 只加载一次 8B 权重(16GB),teardown 回 CPU + empty_cache
(镜像 golden_m0/m1)。
"""

import pytest
import torch

from acorninfer.config import QWEN3_EOS_IDS, EngineConfig
from acorninfer.engine.engine import Engine
from acorninfer.generate import generate_cached
from acorninfer.sampler import SamplingParams
from acorninfer.tokenizer import Tokenizer

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"
SEED = 20261027  # 固定轨迹:fuzz 与其复现完全绑定(失败可重放)
MAX_NEW_TOKENS = 16

# 中英混合、长度各异(12/19/21/22 token)、编码后不含 EOS;按「16 步贪心
# 逐步 top-2 margin 实测」从候选池选出的 headroom 最大组合(证据见 docstring)
PROMPTS = [
    "Mount Everest is the highest mountain above sea level on Earth.",
    "The first three prime numbers are two, three and five, and the next prime after five is seven.",
    "光在真空中的传播速度大约是每秒三十万公里,这是物理学中最基本的常数之一。",
    "水的化学式是H2O,由两个氢原子和一个氧原子组成。",
]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine():
    tok = Tokenizer(f"{QWEN3_8B}/tokenizer.json")
    eng = Engine.from_config(
        # 小池子:max_running 4 强制批处理;512 块对 (128+16)×4 绰绰有余,
        # watermark 2 是策略余量 —— 批处理是常态,这正是被验收的对象
        EngineConfig(model_dir=QWEN3_8B, num_blocks=512, max_running=4, watermark_blocks=2),
        tok,
    )
    yield eng
    eng.assert_balanced()
    eng.model.to("cpu")
    torch.cuda.empty_cache()


def test_golden_m2_concurrent_matches_sequential(engine):
    tok = engine.tokenizer
    prompt_ids = [tok.encode(p) for p in PROMPTS]

    # 参考:M1 的 KV-cache 顺序路径,sampler=None → greedy(argmax)
    refs = [
        generate_cached(engine.model, ids, max_new_tokens=MAX_NEW_TOKENS)[0]
        for ids in prompt_ids
    ]

    # 并发:全部一次提交,temp=0 走 executor 贪心分支(与参考同为 argmax)
    sids = [
        engine.submit(ids, SamplingParams(temperature=0.0), max_tokens=MAX_NEW_TOKENS)
        for ids in prompt_ids
    ]
    while engine.has_pending():
        engine.step()

    # 逐 id 相等,按 seq_id 对号 —— 与完成顺序无关
    for sid, ids, ref in zip(sids, prompt_ids, refs):
        got = engine.seqs[sid].all_ids
        if got != ref:
            i = next(
                (
                    k
                    for k, (a, b) in enumerate(zip(got, ref))
                    if a != b
                ),
                min(len(got), len(ref)),
            )
            raise AssertionError(
                f"token mismatch (engine={got[max(0, i - 2): i + 3]} "
                f"ref={ref[max(0, i - 2): i + 3]}, len {len(got)} vs {len(ref)}): "
                f"first divergence at {i} for prompt {ids[:6]}...; if the true "
                f"top-2 margin at that step is < ~0.25 it is a bf16 tie -> swap "
                f"the prompt per M0 precedent; systematic misalignment = BLOCKED"
            )
        assert got == ref
    engine.assert_balanced()
    # 消费完即释放(engine 契约:drop 由调用方负责),不把残留带进下一个测试
    for sid in sids:
        engine.drop(sid)
    assert not engine.seqs


def test_golden_m2_fuzz_no_leak(engine):
    gen = torch.Generator().manual_seed(SEED)
    vocab = engine.model.cfg.vocab_size

    def rand_ids(n: int) -> list[int]:
        ids = torch.randint(0, vocab, (n,), generator=gen).tolist()
        for eos in QWEN3_EOS_IDS:  # prompt 不含 EOS,防止提交即终止的假阳性
            ids = [(eos + 1) % vocab if t == eos else t for t in ids]
        return ids

    specs = [
        (rand_ids(int(torch.randint(8, 65, (1,), generator=gen))),
         int(torch.randint(4, 17, (1,), generator=gen)))
        for _ in range(20)
    ]

    # 4 波 × 5 条:波间只跑少量 step,下一波 mid-flight 插入(排队 + 批处理 +
    # 抢占机会都真实发生),最后排空
    submitted = 0
    for _ in range(4):
        for _ in range(5):
            ids, max_tokens = specs[submitted]
            engine.submit(ids, SamplingParams(temperature=0.0), max_tokens=max_tokens)
            submitted += 1
        for _ in range(3):
            engine.step()
    while engine.has_pending():
        engine.step()

    assert not engine.has_pending()
    assert len(engine.seqs) == 20
    for seq in engine.seqs.values():
        assert seq.is_finished
        assert seq.stop_reason in ("stop", "length")
        assert 1 <= len(seq.generated_ids) <= 16
    engine.assert_balanced()  # allocator 零 live 块泄漏
    for sid in list(engine.seqs):
        engine.drop(sid)
    assert not engine.seqs
