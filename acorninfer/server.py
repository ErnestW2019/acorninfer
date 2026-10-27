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
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from acorninfer.engine.engine import Engine
from acorninfer.sampler import SamplingParams

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


class PromptCompletionRequest(CompletionRequest):
    """/v1/completions 请求体。"""

    prompt: str


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

    参考规模可接受;真实 BPE 的多字节字符中间态可能停一拍(delta 为空,
    下一个 token 追平)—— 只迟不丢、不重复。
    """

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._ids: list[int] = []
        self._text = ""

    def push(self, token_id: int) -> str:
        self._ids.append(int(token_id))
        text = self._tok.decode(self._ids)
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
    }

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        if state["pump"] is not None:
            state["pump"].cancel()

    app = FastAPI(title="AcornInfer", lifespan=_lifespan)

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
        return seq_id

    def _submit_or_400(req: CompletionRequest, text: str) -> tuple[int, int]:
        """持锁下编码+注册;编码为空或引擎拒绝(容量等)→ 400。返回 (seq_id, prompt_len)。"""
        if state["failure"] is not None:
            raise HTTPException(status_code=500, detail=f"engine failure: {state['failure']}")
        prompt_ids = engine.tokenizer.encode(text)
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="prompt encoded to zero tokens")
        try:
            return _register(req, prompt_ids), len(prompt_ids)
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

    async def _release(seq_id: int) -> None:
        """消费完/断连的收尾:保证每个 finished 序列 exactly-once drop(与 pump GC 互斥)。"""
        async with state["lock"]:
            state["queues"].pop(seq_id, None)
            if state["failure"] is not None:
                state["abandoned"].discard(seq_id)  # 引擎已中毒,不碰其状态
                return
            state["abandoned"].add(seq_id)  # 先标记:防 pump 先 finish 它而漏 GC
            seq = engine.seqs.get(seq_id)
            if seq is not None and seq.is_finished:
                engine.drop(seq_id)
                state["abandoned"].discard(seq_id)

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
        seq_id: int, model: str, created: int, *, chat: bool
    ):
        """SSE:chat 首块带 role → 内容逐块 → 末块空 delta+finish_reason → [DONE]。

        客户端断连时生成器被关闭:不再产出(无 [DONE]),但 finally 仍执行
        _release —— 断连安全,弃管序列由 pump 收尾。
        """
        obj = "chat.completion.chunk" if chat else "text_completion"
        sid = f"chatcmpl-{seq_id}" if chat else f"cmpl-{seq_id}"
        detok = _Detok(engine.tokenizer)
        first = True
        try:
            async for out in _outputs(seq_id):
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
            yield "data: [DONE]\n\n"
        finally:
            await _release(seq_id)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        async with state["lock"]:
            seq_id, prompt_len = _submit_or_400(req, render_chat(req.messages))
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
        async with state["lock"]:
            seq_id, prompt_len = _submit_or_400(req, req.prompt)
        created = int(time.time())
        if req.stream:
            return StreamingResponse(
                _stream_sse(seq_id, req.model, created, chat=False),
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


def main() -> None:
    """生产入口:CLI 参数接 EngineConfig + 真实 tokenizer(测试不经过此路径)。"""
    import argparse

    import uvicorn

    from acorninfer.config import EngineConfig
    from acorninfer.tokenizer import Tokenizer

    parser = argparse.ArgumentParser(prog="acorninfer-server")
    parser.add_argument("--model-dir", default="/home/ernest/model-weight/Qwen3-8B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-running", type=int, default=8)
    parser.add_argument("--watermark-blocks", type=int, default=8)
    args = parser.parse_args()

    cfg = EngineConfig(
        model_dir=args.model_dir,
        device=args.device,
        num_blocks=args.num_blocks,
        page_size=args.page_size,
        max_running=args.max_running,
        watermark_blocks=args.watermark_blocks,
    )
    tok = Tokenizer(f"{args.model_dir}/tokenizer.json")
    engine = Engine.from_config(cfg, tok)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
