"""M3 验收门:triton 后端引擎的并发输出 == 顺序参考(token 逐位)。

协议镜像 test_golden_m2.py:同一批 margin 筛选过的 prompt(golden m2 已记录
margin>~0.25 headroom),双引擎(同一模型对象,属性切换后顺序跑):
  - sdpa 引擎:并发 == 顺序逐位(M2 门回归);
  - triton + fused 引擎:并发 == 顺序逐位(M3 新门);
  - 20 条混合负载 fuzz 跑完 assert_balanced(无泄漏;与 m2 同,不锁 token)。

跨路径语义:顺序参考 generate_cached → model.forward,该路径不读
attention_backend/fused_ops(只有 forward_decode 读)—— 因此无论当前引擎
是哪个后端,参考恒为 sdpa reference;triton 门的差异全部来自引擎的
decode_batch → forward_decode(K2 fused RoPE + K3 分页 attention)。

失误协议:triton 后端若在某 prompt 翻题(bf16 ulp 级平局),按 M0/M2 先例换题
并在此记录新 prompt 的实测 margin —— 系统性错位才是 bug,个别平局不是。
(截至 2026-10-30 首跑:四条 prompt triton 门全数逐位通过,无需换题。)

单卡顺序协议:统一 CUDA_VISIBLE_DEVICES=1;权重只加载一次(module 级
fixture),sdpa/triton 双引擎共享同一模型对象(各建一套分页 cache,约
2×1.2GB,加 16GB 权重 < 24GB),teardown 回 CPU + empty_cache(镜像
golden_m0/m1/m2)。
"""

import pytest
import torch

from acorninfer.config import QWEN3_EOS_IDS
from acorninfer.engine.engine import Engine
from acorninfer.generate import generate_cached
from acorninfer.models.qwen3 import Qwen3ForCausalLM, from_pretrained
from acorninfer.sampler import SamplingParams
from acorninfer.tokenizer import Tokenizer

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"
SEED = 20261027  # 固定轨迹:fuzz 与其复现完全绑定(失败可重放;与 m2 同 seed)
MAX_NEW_TOKENS = 16

# 与 golden m2 完全相同的 4 条 prompt(margin>~0.25 headroom 的实测证据链
# 在 test_golden_m2.py docstring,此处不复制)
PROMPTS = [
    "Mount Everest is the highest mountain above sea level on Earth.",
    "The first three prime numbers are two, three and five, and the next prime after five is seven.",
    "光在真空中的传播速度大约是每秒三十万公里,这是物理学中最基本的常数之一。",
    "水的化学式是H2O,由两个氢原子和一个氧原子组成。",
]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    m = from_pretrained(QWEN3_8B, "cuda:0", torch.bfloat16)
    yield m
    m.to("cpu")
    torch.cuda.empty_cache()


def _make_engine(m: Qwen3ForCausalLM) -> Engine:
    tok = Tokenizer(f"{QWEN3_8B}/tokenizer.json")
    # 512 块 vs 峰值需求 ≤11 块(镜像 golden_m2):批处理常态、抢占结构性不可能
    return Engine(
        m, tok, num_blocks=512, page_size=16, max_running=4, watermark_blocks=2
    )


@pytest.fixture(scope="module")
def sdpa_engine(model):
    eng = _make_engine(model)  # model 默认 attention_backend="sdpa", fused_ops=False
    yield eng
    eng.assert_balanced()


@pytest.fixture(scope="module")
def triton_engine(model):
    # 同一模型对象切属性后重建 Engine(fresh allocator/cache);
    # M1 顺序参考路径不受影响(见 module docstring)
    model.attention_backend = "triton"
    model.fused_ops = True
    eng = _make_engine(model)
    yield eng
    eng.assert_balanced()


def _concurrent_matches_sequential(engine: Engine, label: str) -> None:
    """golden m2 协议:逐请求顺序参考 vs 一次性全量并发提交,逐 id 对号。"""
    tok = engine.tokenizer
    prompt_ids = [tok.encode(p) for p in PROMPTS]

    # 参考:M1 的 KV-cache 顺序路径,sampler=None → greedy(argmax)
    refs = [
        generate_cached(engine.model, ids, max_new_tokens=MAX_NEW_TOKENS)[0]
        for ids in prompt_ids
    ]

    # 并发:全部一次提交,temp=0 走 executor 贪心分支(与参考同为 argmax)
    # spy:钉住批量 decode 真的以 B>1 发生(镜像 m2 —— 引擎若悄悄串行化,
    # forward_decode 独有的 bug 会从本门溜过)
    batch_sizes: list[int] = []
    orig_decode = engine.executor.decode_batch

    def spying_decode(seqs, views):
        batch_sizes.append(len(seqs))
        return orig_decode(seqs, views)

    engine.executor.decode_batch = spying_decode
    sids = [
        engine.submit(ids, SamplingParams(temperature=0.0), max_tokens=MAX_NEW_TOKENS)
        for ids in prompt_ids
    ]
    while engine.has_pending():
        engine.step()
    engine.executor.decode_batch = orig_decode
    assert batch_sizes and max(batch_sizes) >= 2, (
        f"批量 decode 未发生(batch_sizes={batch_sizes}),验收对象落空"
    )

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
                f"[{label}] token mismatch (engine={got[max(0, i - 2): i + 3]} "
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


def test_golden_m3_sdpa_matches_sequential(sdpa_engine):
    assert sdpa_engine.model.attention_backend == "sdpa"
    _concurrent_matches_sequential(sdpa_engine, "sdpa")


def test_golden_m3_triton_matches_sequential(triton_engine):
    # 前置自检:属性必须真在 triton+fused 上,否则本门空转过(退化为重跑 sdpa 门)
    assert triton_engine.model.attention_backend == "triton"
    assert triton_engine.model.fused_ops is True

    # 再钉一层:K3 kernel 真的每步被调(forward_decode 每层调用时才 import,
    # 因此对模块属性重绑定即可 spy;跑完恢复,不污染其他测试)
    import acorninfer.ops.paged_attention as pa

    kernel_calls: list[int] = []
    orig_kernel = pa.paged_decode_attention

    def spying_kernel(*a, **kw):
        kernel_calls.append(1)
        return orig_kernel(*a, **kw)

    pa.paged_decode_attention = spying_kernel
    try:
        _concurrent_matches_sequential(triton_engine, "triton+fused")
    finally:
        pa.paged_decode_attention = orig_kernel
    assert kernel_calls, "K3 paged_decode_attention 未被调用,triton 门验收对象落空"


def test_golden_m3_fuzz_no_leak(triton_engine):
    engine = triton_engine
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

    # 4 波 × 5 条:波间只跑少量 step,下一波 mid-flight 插入(排队 + 批处理
    # 真实发生;池数学下抢占不会触发),最后排空 —— 协议与 m2 fuzz 完全一致
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
