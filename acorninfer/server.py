"""OpenAI 兼容 HTTP server:/v1/chat/completions + /v1/completions(非流式 + SSE 流式)。

并发模型(计划 Task 9 两个选项的落定:专责 pump 任务,懒启动):
- 引擎单线程契约:submit/step/drop 全部在同一把 asyncio.Lock 下串行;CPU 重的
  前向经 asyncio.to_thread 推到线程池,不阻塞事件循环。
- 输出路由:一个专责 pump 任务(首个请求提交时懒启动)是唯一的 step() 驱动方;
  每步把 TokenOutput 按 seq_id 路由进各请求的 asyncio.Queue;当
  `not has_pending()` 且无弃管(abandoned)序列时退出,等下次懒启动 ——
  空闲零开销、不忙轮询事件循环。
- 请求方在 queue.get() 上等待(带空闲超时作 hang 兜底),从不竞争引擎锁。
- drop() 释放契约(Task 8 carry-forward):正常路径由请求方消费完 finish 后
  drop;断连/异常路径在 finally 先标 abandoned 再查 —— 已 finish 则立即 drop,
  未 finish 则由 pump 在路由其收尾输出时代为 drop。锁保证 exactly-once drop,
  engine.seqs 不无界增长。
- engine.step 抛异常 → pump 中毒:记录 failure、把异常推进所有存活队列
  (请求方得到 500)并退出;后续 submit 快速失败 500。
- ChatML 硬编码(运行时不 import transformers/jinja);temperature=0 走
  executor 的贪心分支(OpenAI 语义);SamplingParams 在 pydantic 边界校验
  (temperature>=0、top_p>0、top_k>=0、max_tokens>=1),坏参数到不了引擎。
- 兼容性声明:`max_completion_tokens` 与 `stop` 接受但忽略(声明式子集,
  前者落到默认 max_tokens=16);id 仅进程内唯一;`/v1/completions` 的
  prompt 兼容 token-id 数组(直进引擎不重分词);流式 include_usage 发
  usage 尾包(vLLM parity,bench_remote 协议)。
"""

from __future__ import annotations

import asyncio
import json
import time
import types
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from acorninfer.config import EngineConfig, Qwen3Config
from acorninfer.engine.engine import Engine
from acorninfer.engine.tp_engine import TPEngine, TPLink, install_parent_guard, tp_rank1_loop
from acorninfer.models.qwen3 import from_pretrained, map_hf_state_dict
from acorninfer.models.qwen3_tp import Qwen3TPForCausalLM, build_shard_fn
from acorninfer.sampler import SamplingParams
from acorninfer.tokenizer import Tokenizer
from acorninfer.weights import load_safetensors_sharded

IDLE_TIMEOUT_S = 120.0  # queue.get 空闲兜底:pump 意外消亡时请求不永久挂起


# ---------- 协议模型(我们支持的 OpenAI 子集) ----------


class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    """公共基座:采样/流式字段(/v1/completions 与 /v1/chat/completions 共享)。"""

    model: str = "*"  # 接受并回显,不用(单引擎部署)
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)  # 0 = greedy(executor 分流)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)  # OpenAI 无此字段,vLLM 接受;0 = 关闭
    stream: bool = False


class StreamOptions(BaseModel):
    """OpenAI stream_options 子集:仅 include_usage(流末尾补 usage 块)。"""

    include_usage: bool = False


class PromptCompletionRequest(CompletionRequest):
    """/v1/completions 请求体。

    prompt 双形态(vLLM parity,bench_remote 直投 token-id 数组):str 经
    tokenizer 编码;list[int] 视为已编码 ids 直进引擎(不做重分词)。
    stream_options.include_usage=True 时流末尾追加 choices=[] 的 usage 块
    (vLLM/OpenAI 语义,bench_remote 依赖它读 completion_tokens)。
    """

    prompt: str | list[int]
    stream_options: StreamOptions | None = None


class ChatCompletionRequest(CompletionRequest):
    """/v1/chat/completions 请求体。"""

    messages: list[ChatMessage] = Field(min_length=1)


def render_chat(messages: list[ChatMessage]) -> str:
    """Qwen3 ChatML 硬编码(naive:不做模板注入防护、roles 不校验)。"""
    parts = [f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ---------- 增量解码 / 响应组装 ----------


class _Detok:
    """流式增量解码:每次 decode(至今全部 id),取相对上次的增量后缀。

    真实 ByteLevel 解码对跨 token 切碎的多字节字符会先产出 U+FFFD 中间态;
    中间态一律扣发(不发、不推进基线),拼完整后再发 —— 否则客户端收到
    \\ufffd 且真实字符会被对齐逻辑吞掉。代价:流末尾恰为孤立坏字节时该字符
    不发(合法 UTF-8 流不触发)。参考规模可接受;只迟不丢、不重复。
    """

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._ids: list[int] = []
        self._text = ""

    def push(self, token_id: int) -> str:
        self._ids.append(int(token_id))
        text = self._tok.decode(self._ids)
        if text.endswith("\ufffd"):
            return ""  # 中间态:扣发
        delta = text[len(self._text) :] if text.startswith(self._text) else ""
        self._text = text
        return delta


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# ---------- app 工厂 ----------


def create_app(engine: Engine) -> FastAPI:
    if engine.tokenizer is None:
        raise ValueError(
            "create_app 需要带 tokenizer 的引擎(server 侧编解码边界依赖它)"
        )
    state = {
        "lock": asyncio.Lock(),
        "queues": {},  # seq_id -> asyncio.Queue[TokenOutput | Exception]
        "abandoned": set(),  # 已弃管未收尾的 seq_id(断连路径,pump 负责收尾 drop)
        "pump": None,  # asyncio.Task | None
        "failure": None,  # 引擎级故障(中毒标记)
        "reapers": set(),  # 独立收尾任务的强引用(防被 GC)
    }

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        if state["pump"] is not None:
            state["pump"].cancel()

    app = FastAPI(title="AcornInfer", lifespan=_lifespan)

    def _on_pump_done(task: asyncio.Task) -> None:
        """pump 非正常死亡(路由循环 bug 等)必须可见:标 failure 并唤醒等待者。

        中毒路径在 _pump 内已自行设 failure+通知,此处只兜"没接住的异常"。
        """
        if task.cancelled():  # lifespan 关停的正常取消
            return
        exc = task.exception()
        if exc is None:
            return
        if state["failure"] is None:
            state["failure"] = exc
        for q in state["queues"].values():
            q.put_nowait(exc)
        state["queues"].clear()

    async def _pump() -> None:
        """唯一 step 驱动方:锁内 查 pending → to_thread(step) → 路由/GC;空闲即退。"""
        while True:
            async with state["lock"]:
                if not state["abandoned"] and not engine.has_pending():
                    return
                try:
                    outputs = await asyncio.to_thread(engine.step)
                except Exception as exc:  # 引擎级故障:通知所有等待者后退出
                    state["failure"] = exc
                    for q in state["queues"].values():
                        q.put_nowait(exc)
                    state["queues"].clear()
                    state["abandoned"].clear()
                    return
                for out in outputs:
                    q = state["queues"].get(out.seq_id)
                    if q is not None:
                        q.put_nowait(out)
                    elif out.finished and out.seq_id in state["abandoned"]:
                        engine.drop(out.seq_id)  # 弃管序列 finish 即收尾
                        state["abandoned"].discard(out.seq_id)

    def _register(req: CompletionRequest, prompt_ids: list[int]) -> int:
        """调用方必须已持锁:submit → 建队列 → 懒启动 pump(防丢唤醒)。"""
        seq_id = engine.submit(
            prompt_ids,
            SamplingParams(
                temperature=req.temperature, top_p=req.top_p, top_k=req.top_k
            ),
            max_tokens=req.max_tokens,
        )
        state["queues"][seq_id] = asyncio.Queue()
        if state["pump"] is None or state["pump"].done():
            state["pump"] = asyncio.create_task(_pump(), name="engine-pump")
            state["pump"].add_done_callback(_on_pump_done)
        return seq_id

    def _register_or_400(req: CompletionRequest, prompt_ids: list[int]) -> int:
        """调用方必须已持锁:failure 检查 → submit → 建队列 → 懒启动 pump(防丢唤醒)。

        引擎拒绝(容量等)→ 400。编码在锁外完成(与引擎状态无关,长 prompt
        不拖累 pump)。
        """
        if state["failure"] is not None:
            raise HTTPException(status_code=500, detail=f"engine failure: {state['failure']}")
        try:
            return _register(req, prompt_ids)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def _outputs(seq_id: int):
        """逐个产出 TokenOutput 直到 finished;引擎故障 → 500;空闲超时 → 504。"""
        q = state["queues"].get(seq_id)
        if q is None:  # 仅在中毒清队列后可能发生
            raise HTTPException(status_code=500, detail="engine failure: queue lost")
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=IDLE_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                raise HTTPException(status_code=504, detail="engine idle timeout") from exc
            if isinstance(item, Exception):
                raise HTTPException(status_code=500, detail=f"engine failure: {item}") from item
            yield item
            if item.finished:
                return

    def _release_locked(seq_id: int) -> None:
        """持锁收尾体:弹队列、标弃管、已 finish 则 drop(与 pump GC 互斥)。"""
        state["queues"].pop(seq_id, None)
        if state["failure"] is not None:
            state["abandoned"].discard(seq_id)  # 引擎已中毒,不碰其状态
            return
        state["abandoned"].add(seq_id)  # 先标记:防 pump 先 finish 它而漏 GC
        seq = engine.seqs.get(seq_id)
        if seq is not None and seq.is_finished:
            engine.drop(seq_id)
            state["abandoned"].discard(seq_id)

    def _spawn_reaper(seq_id: int) -> None:
        task = asyncio.create_task(_reap(seq_id))
        state["reapers"].add(task)
        task.add_done_callback(state["reapers"].discard)

    async def _reap(seq_id: int) -> None:
        """独立收尾任务:不受请求协程的取消影响,拿到锁即完成清理。"""
        async with state["lock"]:
            _release_locked(seq_id)

    async def _release(seq_id: int) -> None:
        """消费完/断连的收尾:每个 finished 序列 exactly-once drop。

        断连取消的杀伤点:uvicorn(ASGI spec 2.3)经 anyio collapsing 取消
        流式任务,取消会在每个悬挂点反复重投递;而 pump 持锁跨整个前向 ——
        清理若在此悬挂,会被反复取消而死不成,seq+queue 永久泄漏。对策:
        锁被占时直接转交独立 reap 任务;锁空闲时内联执行(消费完流 → seqs
        已空,测试的确定性依赖它)。若在悬挂点被取消,同样转交 reap。
        """
        if state["lock"].locked():
            _spawn_reaper(seq_id)
            return
        try:
            async with state["lock"]:
                _release_locked(seq_id)
        except asyncio.CancelledError:
            _spawn_reaper(seq_id)

    async def _collect(seq_id: int) -> tuple[list[int], str | None]:
        """非流式:收齐全部输出;释放放调用方 finally(异常路径也要释放)。"""
        ids: list[int] = []
        reason = None
        async for out in _outputs(seq_id):
            ids.append(out.token_id)
            if out.finished:
                reason = out.stop_reason
        return ids, reason

    def _chunk(
        sid: str,
        obj: str,
        model: str,
        created: int,
        *,
        chat: bool,
        role: str | None = None,
        delta: str | None = None,
        finish_reason: str | None = None,
    ) -> dict:
        choice: dict = {"index": 0, "finish_reason": finish_reason}
        if chat:
            d: dict = {}
            if role is not None:
                d["role"] = role
            if delta:
                d["content"] = delta
            choice["delta"] = d
        else:
            choice["text"] = delta or ""
        return {
            "id": sid,
            "object": obj,
            "created": created,
            "model": model,
            "choices": [choice],
        }

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _stream_sse(
        seq_id: int, model: str, created: int, *,
        chat: bool, prompt_len: int = 0, include_usage: bool = False,
    ):
        """SSE:chat 首块带 role → 内容逐块 → 末块空 delta+finish_reason
        → (include_usage 时)choices=[] 的 usage 块 → [DONE]。

        客户端断连时生成器被关闭:不再产出(无 [DONE]),但 finally 仍执行
        _release —— 断连安全,弃管序列由 pump 收尾。
        """
        obj = "chat.completion.chunk" if chat else "text_completion"
        sid = f"chatcmpl-{seq_id}" if chat else f"cmpl-{seq_id}"
        detok = _Detok(engine.tokenizer)
        first = True
        n = 0
        try:
            async for out in _outputs(seq_id):
                n += 1
                delta = detok.push(out.token_id)
                if first and chat:
                    first = False
                    yield _sse(
                        _chunk(sid, obj, model, created, chat=True, role="assistant")
                    )
                if delta:
                    yield _sse(
                        _chunk(sid, obj, model, created, chat=chat, delta=delta)
                    )
                if out.finished:
                    yield _sse(
                        _chunk(
                            sid, obj, model, created, chat=chat,
                            finish_reason=out.stop_reason,
                        )
                    )
            if include_usage:
                yield _sse(
                    {
                        "id": sid,
                        "object": obj,
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": _usage(prompt_len, n),
                    }
                )
            yield "data: [DONE]\n\n"
        finally:
            await _release(seq_id)

    @app.get("/health")
    def health():
        if state["failure"] is not None:
            # 中毒后必须可见,让监控/编排在位重启,而不是永远返回绿色
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=503,
                content={"ok": False, "error": str(state["failure"])},
            )
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        prompt_ids = engine.tokenizer.encode(render_chat(req.messages))
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="prompt encoded to zero tokens")
        prompt_len = len(prompt_ids)
        async with state["lock"]:
            seq_id = _register_or_400(req, prompt_ids)
        created = int(time.time())
        if req.stream:
            return StreamingResponse(
                _stream_sse(seq_id, req.model, created, chat=True),
                media_type="text/event-stream",
            )
        try:
            ids, reason = await _collect(seq_id)
        finally:
            await _release(seq_id)
        return {
            "id": f"chatcmpl-{seq_id}",
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": engine.tokenizer.decode(ids)},
                    "finish_reason": reason,
                }
            ],
            "usage": _usage(prompt_len, len(ids)),
        }

    @app.post("/v1/completions")
    async def completions(req: PromptCompletionRequest):
        if isinstance(req.prompt, str):
            prompt_ids = engine.tokenizer.encode(req.prompt)
        else:  # list[int]:已编码 ids 直投(vLLM parity),不做重分词
            prompt_ids = req.prompt
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="prompt encoded to zero tokens")
        prompt_len = len(prompt_ids)
        async with state["lock"]:
            seq_id = _register_or_400(req, prompt_ids)
        created = int(time.time())
        if req.stream:
            return StreamingResponse(
                _stream_sse(
                    seq_id, req.model, created, chat=False, prompt_len=prompt_len,
                    include_usage=bool(
                        req.stream_options and req.stream_options.include_usage
                    ),
                ),
                media_type="text/event-stream",
            )
        try:
            ids, reason = await _collect(seq_id)
        finally:
            await _release(seq_id)
        return {
            "id": f"cmpl-{seq_id}",
            "object": "text_completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "text": engine.tokenizer.decode(ids),
                    "finish_reason": reason,
                }
            ],
            "usage": _usage(prompt_len, len(ids)),
        }

    return app


def _prefix_normalized_shard_fn(cfg, rank: int, world: int):
    """装载边界键归一(与单卡 from_pretrained 的 map_hf_state_dict 同口径)。

    checkpoint 两套键型:HF 原生解码器参数带 "model." 前缀(8B 实测,
    lm_head 裸名),仓库自存 tiny ckpt 无前缀;qwen3_tp.build_shard_fn 的
    匹配键为无前缀名(embed/lm_head/norm 精确名匹配,层名 endswith)。
    剥离前缀后转发;load_safetensors_sharded 返回的 state 亦经
    map_hf_state_dict 归一成仓库平铺键再 strict 装入。归一层驻 server
    (生产 shard_fn 调用点),qwen3_tp.py 零改动。
    """
    inner = build_shard_fn(cfg, rank, world)

    def shard_fn(name: str, t, r: int, w: int):
        return inner(name.removeprefix("model."), t, r, w)

    return shard_fn


def _build_tp_model(model_dir: str, rank: int, world: int, device: str, dtype):
    """TP 模型构造(生产路径;测试经 monkeypatch 打桩):
    config 读 checkpoint 同源 → CPU 全量载 + 结构化分片(HF "model." 前缀
    经 _prefix_normalized_shard_fn 归一,两套键型通吃)→ strict assign 装入
    → 单次 .to(峰值显存 = 一份分片,与 from_pretrained 口径一致)。"""
    cfg = Qwen3Config.from_json(f"{model_dir}/config.json")
    state = load_safetensors_sharded(
        model_dir, _prefix_normalized_shard_fn(cfg, rank, world),
        rank, world, device="cpu",
    )
    state = map_hf_state_dict(state)
    model = Qwen3TPForCausalLM(cfg, rank, world)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval()


def _build_engine(args, link: TPLink | None = None):
    """从解析后的 CLI 参数构造 Engine(生产路径;测试经 monkeypatch 打桩)。

    --tp 1(缺省):与 M5 Task 2 逐字节同一(engine.py 的 from_config 尚无
    chunked_budget 透传,故内联;两处构造需保持同步):
    - attention_backend/fused_ops/decode_graph 在 Engine 构造前置位(decode
      graph runner 在 Engine.__init__ 捕获各桶);后端配方旗标
      --attention-backend/--fused-ops/--cuda-graph(bench 同名)透传
      EngineConfig,缺省 sdpa/无 fused/无 graph = Task 2 行为逐字节;
    - chunked_budget 经 Engine 构造参数透传(M5 chunked prefill),缺省
      None = 每步全量 prefill,引擎行为与 M4.1 逐字节一致。
    --tp >= 2:分片 TP 模型(HF "model." 前缀经 _prefix_normalized_shard_fn
    归一)+ Engine(tp=link);rank0(link.rank==0)外罩 TPEngine 命令壳返回。
    rank1 由 _tp_rank1_main 经本函数以 TPLink(1) 构造(返回裸 Engine,无壳)。
    """
    tp = getattr(args, "tp", 1)
    # 后端配方旗标(--attention-backend/--fused-ops/--cuda-graph,bench 同名;
    # getattr 缺省 = CLI 缺省 = 旧调用方(SimpleNamespace 无字段)零改动;
    # --cuda-graph 须 triton 由 EngineConfig.__post_init__ 构造期校验)
    backend = dict(
        attention_backend=getattr(args, "attention_backend", "sdpa"),
        fused_ops=getattr(args, "fused_ops", False),
        decode_graph=getattr(args, "cuda_graph", False),
    )
    if tp == 1:
        cfg = EngineConfig(
            model_dir=args.model_dir,
            device=args.device,
            num_blocks=args.num_blocks,
            page_size=args.page_size,
            max_running=args.max_running,
            watermark_blocks=args.watermark_blocks,
            **backend,
        )
        model = from_pretrained(cfg.model_dir, cfg.device, cfg.dtype)
        model.attention_backend = cfg.attention_backend
        model.fused_ops = cfg.fused_ops
        model.decode_graph = cfg.decode_graph
        tok = Tokenizer(f"{args.model_dir}/tokenizer.json")
        return Engine(
            model,
            tok,
            num_blocks=cfg.num_blocks,
            page_size=cfg.page_size,
            max_running=cfg.max_running,
            watermark_blocks=cfg.watermark_blocks,
            chunked_budget=args.chunked_budget,
        )
    assert link is not None, "--tp >= 2 需要 link(TPLink(rank))"
    cfg = EngineConfig(
        model_dir=args.model_dir,
        device=args.device,
        num_blocks=args.num_blocks,
        page_size=args.page_size,
        max_running=args.max_running,
        watermark_blocks=args.watermark_blocks,
        **backend,
    )
    # 设备放置(根因修复,M5T4 审查 Blocker):TP 下每 rank 必须用
    # f"cuda:{rank}"(CUDA_VISIBLE_DEVICES 序数映射,与全部测试/worker 同一口径)。
    # 曾用 cfg.device(--device,TP=1 语义)给两 rank —— rank1 的模型/激活
    # 落 GPU0 而 NCCL comm 绑 GPU1(set_device(rank)),comm/张量跨卡脑裂:
    # 首次 prefill 的 allreduce 永不配对,rank0 宿主线程楔死在 libcuda
    # (cuLibraryGetGlobal);nvidia-smi 见 rank1 双卡占显存、NCCL INFO 见
    # rank0 comm cudaDev0/rank1 comm cudaDev1。--device 在 --tp>=2 下不生效。
    model = _build_tp_model(args.model_dir, link.rank, tp,
                            f"cuda:{link.rank}", cfg.dtype)
    model.attention_backend = cfg.attention_backend
    model.fused_ops = cfg.fused_ops
    model.decode_graph = cfg.decode_graph
    tok = Tokenizer(f"{args.model_dir}/tokenizer.json") if link.rank == 0 else None
    engine = Engine(
        model,
        tok,
        num_blocks=cfg.num_blocks,
        page_size=cfg.page_size,
        max_running=cfg.max_running,
        watermark_blocks=cfg.watermark_blocks,
        chunked_budget=args.chunked_budget,
        tp=link,
    )
    if link.rank == 0:
        return TPEngine(engine, link)
    return engine


def _tp_rank1_main(args_dict: dict, init_dir: str) -> None:
    """server rank1 入口(spawn 子进程):孤儿防护 → NCCL → 同构 engine →
    命令循环。rank0 死 → 本进程必死(install_parent_guard);FINI → 优雅收尾。"""
    install_parent_guard(poll_s=5.0)
    import os

    from acorninfer.distributed import destroy, setup

    args = types.SimpleNamespace(**args_dict)
    setup(1, args.tp, f"file://{init_dir}/nccl")
    try:
        link = TPLink(1)
        engine = _build_engine(args, link=link)
        tp_rank1_loop(engine, link)
    finally:
        # destroy 收尾有时挂(T0 先例):限 10s,挂了硬退兜底 —— 显存必放,
        # 不给 T6 的 PID kill 纪律新增僵尸类
        import threading

        t = threading.Thread(target=destroy, daemon=True)
        t.start()
        t.join(timeout=10.0)
    os._exit(0)


def main() -> None:
    """生产入口:CLI 参数接 EngineConfig + 真实 tokenizer(测试不经过此路径)。

    --tp 2:父进程 = rank0(持 FastAPI),spawn 子进程 rank1(同参 + file://
    init);rank0 退出路径(uvicorn 返回/异常)广播 FINI → join → 兜底硬杀,
    不留孤儿。
    """
    import argparse
    import multiprocessing as mp
    import tempfile

    import uvicorn

    parser = argparse.ArgumentParser(prog="acorninfer-server")
    parser.add_argument("--model-dir", default="/home/ernest/model-weight/Qwen3-8B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-running", type=int, default=8)
    parser.add_argument("--watermark-blocks", type=int, default=8)
    parser.add_argument(
        "--chunked-budget", type=int, default=None,
        help="每步 prefill token 预算(M5 chunked prefill);缺省 None = 不切块",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="张量并行度(M5 Task 4);缺省 1 = 单卡逐字节现行为",
    )
    parser.add_argument(
        "--attention-backend", default="sdpa", choices=["sdpa", "triton"],
        help="attention 后端(M3 paged attention;与 bench 同名旗标)",
    )
    parser.add_argument(
        "--fused-ops", action="store_true",
        help="decode 走 K1/K2 融合算子(M3)",
    )
    parser.add_argument(
        "--cuda-graph", action="store_true",
        help="decode 走 CUDA graph(M4;须 --attention-backend triton)",
    )
    args = parser.parse_args()

    if args.tp == 1:
        engine = _build_engine(args)
        uvicorn.run(create_app(engine), host=args.host, port=args.port)
        return

    init_dir = tempfile.mkdtemp(prefix="acorninfer-tp-")
    ctx = mp.get_context("spawn")
    child = ctx.Process(target=_tp_rank1_main, args=(vars(args), init_dir))
    child.start()
    from acorninfer.distributed import destroy, setup

    engine = None
    try:
        setup(0, args.tp, f"file://{init_dir}/nccl")
        engine = _build_engine(args, link=TPLink(0))
        uvicorn.run(create_app(engine), host=args.host, port=args.port)
    finally:
        if engine is not None:
            engine.shutdown()      # FINI 广播:rank1 优雅收尾
        child.join(timeout=30)
        if child.is_alive():
            child.kill()           # 兜底硬杀:不许孤儿(T6 PID kill 纪律依赖)
            child.join(timeout=10)
        destroy()


if __name__ == "__main__":
    main()
