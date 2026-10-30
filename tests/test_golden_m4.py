"""M4 验收门:graph 后端引擎的并发输出 == 顺序参考(token 逐位)。

协议镜像 test_golden_m3.py:同 margin 筛选 prompt,三引擎(同一模型对象,
属性切换后各建 Engine):
  - sdpa:回归门(与 m3 同);
  - triton eager:回归门(与 m3 同);
  - triton graph:M4 新门 —— decode_graph=True 须在 Engine 构造前置位;
  - fuzz no-leak:graph 开,跑完 assert_balanced。
失误协议:graph 门若翻题按 M0/M2/M3 先例换题并记录新 margin;系统性错位才是 bug。
(截至 2026-10-31 首跑:四条 prompt 三门全数逐位通过,无需换题。)

跨路径语义(镜像 m3):顺序参考 generate_cached → model.forward,该路径不读
attention_backend/fused_ops/decode_graph(只有 forward_decode 读)—— 因此
无论当前引擎是哪个后端,参考恒为 sdpa reference;graph 门的差异全部来自
Engine 的 decode_batch 分流(triton 后端 + 全 greedy + B ≤ 8 时走
DecodeGraphRunner 静态图重放,否则 eager forward_decode)。注意:replay
不执行图内 python —— K3 kernel 只在构造期捕获时跑,因此 graph 门的防空转
钉钉在 runner 分发层而非 kernel 层(m3 钉 kernel 的手法在 graph 门失效)。

decode_graph 前置契约:DecodeGraphRunner 在 Engine.__init__ 里构造并按桶
{1,2,4,8} 捕获;model.decode_graph 必须在 Engine(...) 之前置位 —— 构造后
再置位 runner 不存在,Executor 永远走 eager,图门空转(第一钉查这个)。

单卡顺序协议:统一 CUDA_VISIBLE_DEVICES=1;权重只加载一次(module 级
fixture),三引擎共享同一模型对象(各建一套分页 cache,3×~1.2GB,加
16GB 权重 < 24GB),teardown 回 CPU + empty_cache(镜像 golden_m3)。
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
SEED = 20261027  # 固定轨迹:fuzz 与其复现完全绑定(失败可重放;与 m2/m3 同 seed)
MAX_NEW_TOKENS = 16

# 与 golden m2/m3 完全相同的 4 条 prompt(margin>~0.25 headroom 的实测证据链
# 在 test_golden_m2.py docstring,此处不复制)。禁止改集:改了三门结论就与
# m2/m3 不可比;graph 门若翻题,按先例换题并在 module docstring 记录新 margin。
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
    # 512 块 vs 峰值需求 ≤11 块(镜像 golden_m2/m3):批处理常态、抢占结构性不可能
    return Engine(
        m, tok, num_blocks=512, page_size=16, max_running=4, watermark_blocks=2
    )


@pytest.fixture(scope="module")
def sdpa_engine(model):
    # M4 有三个可切属性(backend/fused_ops/decode_graph),全部显式归零:
    # 不靠"默认值 + 前序 fixture 收尾复位"的隐式顺序,防串门
    model.attention_backend = "sdpa"
    model.fused_ops = False
    model.decode_graph = False
    eng = _make_engine(model)
    assert eng.graph_runner is None  # 开关归零 → 不建 runner(镜像 m3 前置自检风格)
    yield eng
    eng.assert_balanced()


@pytest.fixture(scope="module")
def triton_engine(model):
    # 同一模型对象切属性后重建 Engine(fresh allocator/cache);
    # M1 顺序参考路径不受影响(见 module docstring)。
    # decode_graph 显式 False:本门验收 eager triton,runner 不得存在
    model.attention_backend = "triton"
    model.fused_ops = True
    model.decode_graph = False
    eng = _make_engine(model)
    assert eng.graph_runner is None
    yield eng
    eng.assert_balanced()
    # 复位后端属性:triton 阶段结束不留全局状态(镜像 m3)
    model.attention_backend = "sdpa"
    model.fused_ops = False
    model.decode_graph = False


@pytest.fixture(scope="module")
def graph_engine(model):
    # M4 新门:decode_graph=True 必须在 Engine(...) 构造之前置位 —— runner
    # 在 __init__ 捕获各桶,构造后再置位 = runner 不存在 = 门空转
    model.attention_backend = "triton"
    model.fused_ops = True
    model.decode_graph = True
    assert model.decode_graph is True  # 前置自检:置位先于构造
    eng = _make_engine(model)
    # 防空转第一钉:开关真的生效(否则本门退化成重跑 eager 门)
    assert eng.graph_runner is not None
    yield eng
    eng.assert_balanced()  # graph 分支:排空后恰 1 块 live(runner 常驻 scratch)
    # 复位:graph 阶段结束不留全局状态(镜像 m3 triton fixture 的收尾复位)
    model.attention_backend = "sdpa"
    model.fused_ops = False
    model.decode_graph = False


def _concurrent_matches_sequential(engine: Engine, label: str) -> None:
    """golden m2/m3 协议:逐请求顺序参考 vs 一次性全量并发提交,逐 id 对号。"""
    tok = engine.tokenizer
    prompt_ids = [tok.encode(p) for p in PROMPTS]

    # 参考:M1 的 KV-cache 顺序路径,sampler=None → greedy(argmax)
    refs = [
        generate_cached(engine.model, ids, max_new_tokens=MAX_NEW_TOKENS)[0]
        for ids in prompt_ids
    ]

    # 并发:全部一次提交,temp=0 走 executor 贪心分支(与参考同为 argmax)
    # spy:钉住批量 decode 真的以 B>1 发生(镜像 m2/m3 —— 引擎若悄悄串行化,
    # 批量路径独有的 bug 会从本门溜过)
    batch_sizes: list[int] = []
    orig_decode = engine.executor.decode_batch

    def spying_decode(seqs, views):
        batch_sizes.append(len(seqs))
        return orig_decode(seqs, views)

    engine.executor.decode_batch = spying_decode
    try:
        sids = [
            engine.submit(
                ids, SamplingParams(temperature=0.0), max_tokens=MAX_NEW_TOKENS
            )
            for ids in prompt_ids
        ]
        while engine.has_pending():
            engine.step()
    finally:
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


def test_golden_m4_sdpa_matches_sequential(sdpa_engine):
    assert sdpa_engine.model.attention_backend == "sdpa"
    assert sdpa_engine.model.decode_graph is False
    _concurrent_matches_sequential(sdpa_engine, "sdpa")


def test_golden_m4_triton_eager_matches_sequential(triton_engine):
    # 前置自检:属性必须真在 triton+fused 上,否则本门空转过(退化为重跑 sdpa 门)
    assert triton_engine.model.attention_backend == "triton"
    assert triton_engine.model.fused_ops is True
    assert triton_engine.model.decode_graph is False

    # 再钉一层:K3 kernel 真的每步被调(forward_decode 每层调用时才 import,
    # 因此对模块属性重绑定即可 spy;跑完恢复,不污染其他测试;镜像 m3)
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


def test_golden_m4_graph_matches_sequential(graph_engine):
    engine = graph_engine
    # 门内再核属性(fixture 已断言构造期两钉):防 fixture 串门
    assert engine.model.attention_backend == "triton"
    assert engine.model.fused_ops is True
    assert engine.model.decode_graph is True

    # 防空转钉(graph 版):m3 钉 K3 kernel 的手法在 graph 门失效 —— replay
    # 不执行图内 python,kernel 每步并不被调。改钉分发层,两向夹住:
    # 1) runner.decode_batch 真的被 Executor 分流调用(且批量 B≥2);
    # 2) 门内 model.forward_decode 零调用 —— eager 回退必经它,而重放不执行
    #    python;计数 > 0 即 graph 分流落空,本门验收对象错了。
    rb_calls: list[int] = []
    orig_rb = engine.graph_runner.decode_batch

    def spying_rb(seqs, views):
        rb_calls.append(len(seqs))
        return orig_rb(seqs, views)

    engine.graph_runner.decode_batch = spying_rb

    fd_calls: list[int] = []
    orig_fd = engine.model.forward_decode

    def spying_fd(*a, **kw):
        fd_calls.append(1)
        return orig_fd(*a, **kw)

    engine.model.forward_decode = spying_fd
    try:
        _concurrent_matches_sequential(engine, "triton+graph")
    finally:
        engine.graph_runner.decode_batch = orig_rb
        del engine.model.forward_decode  # 撤实例属性遮蔽,恢复类方法

    assert rb_calls, "graph decode_batch 未被调用,graph 门验收对象落空"
    assert max(rb_calls) >= 2, f"graph 分流只见过单批(rb_calls={rb_calls})"
    assert not fd_calls, (
        f"门内 eager forward_decode 被调 {len(fd_calls)} 次 —— decode 未走 graph 重放"
    )
    # 第三钉:确曾 replay(last_logits 由 replay 后原址覆写,捕获时为 None)
    assert engine.graph_runner.last_logits is not None


def test_golden_m4_fuzz_no_leak(graph_engine):
    engine = graph_engine
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
    # 真实发生;批 ≤ max_running=4 ≤ 8,每步 decode 都落 graph 重放;池数学下
    # 抢占不会触发),最后排空 —— 协议与 m2/m3 fuzz 完全一致
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
    engine.assert_balanced()  # graph 契约:allocator 恰 1 块 live(scratch),余即泄漏
    for sid in list(engine.seqs):
        engine.drop(sid)
    assert not engine.seqs
