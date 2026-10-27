"""OpenAI 兼容 server 合同测试(tiny 引擎,httpx ASGI 进程内,无网络/uvicorn)。

覆盖:非流式 chat 并发、流式 SSE 拼接 == 非流式、completions 往返、
请求校验 422、4 路流式并发 == 贪心参考、drop() 释放契约
(每个测试结束后 engine 无残留 seqs、块平衡)。
"""

import asyncio
import json

import pytest
import torch
from httpx import ASGITransport, AsyncClient

from acorninfer.engine.engine import Engine
from acorninfer.generate import generate_cached
from acorninfer.server import ChatMessage, create_app, render_chat

pytestmark = pytest.mark.anyio  # anyio 自带 pytest 插件,零新依赖(asyncio 后端)


class FakeTok:
    """mod-256 tokenizer:encode 后的 id 必落 tiny 256 词表内;
    decode 对任意 id 列表全,total、永不抛(任务硬性要求)。"""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(int(i) % 256) for i in ids)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def engine(tiny_model):
    eng = Engine(
        tiny_model,
        tokenizer=FakeTok(),
        num_blocks=64,
        page_size=4,
        max_running=4,
        watermark_blocks=0,
    )
    yield eng
    # drop() 契约:每个请求消费完即释放;引擎不得残留 seqs,块必须平衡
    eng.assert_balanced()
    assert not eng.seqs


@pytest.fixture
async def client(engine):
    async with AsyncClient(
        transport=ASGITransport(create_app(engine)), base_url="http://test"
    ) as c:
        yield c


async def _chat_sse_chunks(client, payload) -> list[dict]:
    """POST 流式 chat 请求,解析 SSE:返回 [DONE] 前的 chunk 列表。"""
    async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [line async for line in r.aiter_lines()]
    data = [ln[len("data: ") :] for ln in lines if ln.startswith("data: ")]
    assert data, "no SSE data lines"
    assert data[-1] == "[DONE]"
    return [json.loads(d) for d in data[:-1]]


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


async def test_chat_non_streaming_two_concurrent(client, engine):
    msgs = [ChatMessage(role="user", content="hello")]
    payload = {
        "model": "acorn-tiny",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    r1, r2 = await asyncio.gather(
        client.post("/v1/chat/completions", json=payload),
        client.post("/v1/chat/completions", json=payload),
    )
    assert r1.status_code == r2.status_code == 200
    prompt_len = len(FakeTok().encode(render_chat(msgs)))
    texts = []
    for r in (r1, r2):
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "acorn-tiny"  # model 字段回显
        assert isinstance(body["created"], int)
        choice = body["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "length"  # tiny 无可达 EOS → 必走 max_tokens
        assert choice["message"]["role"] == "assistant"
        usage = body["usage"]
        assert usage["prompt_tokens"] == prompt_len
        assert usage["completion_tokens"] == 4
        assert usage["total_tokens"] == prompt_len + 4
        texts.append(choice["message"]["content"])
    assert texts[0] == texts[1]  # 同 prompt 贪心 → 批处理不改结果
    engine.assert_balanced()


async def test_chat_streaming_matches_non_streaming(client):
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 6,
        "temperature": 0.0,
    }
    ref = (await client.post("/v1/chat/completions", json=payload)).json()[
        "choices"
    ][0]["message"]["content"]
    chunks = await _chat_sse_chunks(client, {**payload, "stream": True})
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    first = chunks[0]["choices"][0]
    assert first["delta"].get("role") == "assistant"  # 首块带 role
    assert first["finish_reason"] is None
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == ref  # 流式拼接 == 非流式(贪心)
    last = chunks[-1]["choices"][0]
    assert last["finish_reason"] == "length"
    assert last["delta"].get("content", "") == ""  # 末块空 delta 只带 finish_reason
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


async def test_completions_roundtrip(client, engine):
    r = await client.post(
        "/v1/completions", json={"prompt": "abc", "max_tokens": 5, "temperature": 0.0}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "length"
    assert isinstance(body["choices"][0]["text"], str)
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    engine.assert_balanced()


async def test_validation_422_engine_untouched(client, engine):
    cases = [
        ("/v1/completions", {"prompt": "x", "temperature": -1.0, "max_tokens": 2}),
        ("/v1/completions", {"prompt": "x", "max_tokens": 0}),
        ("/v1/completions", {"prompt": "x", "top_p": 0.0}),
        ("/v1/completions", {"prompt": "x", "top_k": -1}),
        ("/v1/chat/completions", {"messages": []}),
    ]
    for url, payload in cases:
        r = await client.post(url, json=payload)
        assert r.status_code == 422, (url, payload)
    assert not engine.has_pending()  # 引擎未被坏请求触碰
    assert not engine.seqs


async def test_empty_prompt_400_then_recovery(client, engine):
    r = await client.post("/v1/completions", json={"prompt": "", "max_tokens": 2})
    assert r.status_code == 400
    r = await client.post("/v1/completions", json={"prompt": "still alive", "max_tokens": 2})
    assert r.status_code == 200  # 坏请求后服务正常
    assert not engine.seqs


async def test_four_concurrent_streams_match_greedy_reference(client, engine, tiny_model):
    """4 路同时流式:全部完成,输出逐位等于单序列贪心参考(generate_cached)。"""
    tok = FakeTok()
    prompts = ["the ", "qwen", "abc", "zz-zz"]
    n = 6
    refs = []
    with torch.inference_mode():
        for p in prompts:
            ids = tok.encode(p)
            out, _ = generate_cached(tiny_model, ids, max_new_tokens=n)
            refs.append(tok.decode(out[len(ids) :]))

    async def one(payload):
        async with client.stream("POST", "/v1/completions", json=payload) as r:
            assert r.status_code == 200
            lines = [ln async for ln in r.aiter_lines()]
        data = [ln[len("data: ") :] for ln in lines if ln.startswith("data: ")]
        assert data[-1] == "[DONE]"
        chunks = [json.loads(d) for d in data[:-1]]
        assert all(c["object"] == "text_completion" for c in chunks)
        return "".join(c["choices"][0]["text"] for c in chunks)

    payloads = [
        {"prompt": p, "max_tokens": n, "temperature": 0.0, "stream": True}
        for p in prompts
    ]
    texts = await asyncio.gather(*(one(pl) for pl in payloads))
    assert list(texts) == refs
    engine.assert_balanced()
    assert not engine.seqs
