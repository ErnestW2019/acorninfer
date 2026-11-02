"""M9 Task 0 R5 取证代理:转发 HTTP 到目标 server 并逐请求落盘请求体哈希。

跑法(压测侧,CPU):
  .venv/bin/python scripts/m9_capture_proxy.py --listen 8901 --target \
      http://127.0.0.1:8100 --log /tmp/opencode/m9t0/r5_vllm.jsonl

用途:官方 vllm bench serve 首跑(同 seed)经代理分别打两侧 server,取两侧
收到的 prompts 哈希序列对比(R5:同 tokenizer+同 seed → 同 prompts)。
响应原样流式转发(SSE 透传);代理仅用于取证短跑,不入成绩单。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time

import aiohttp
from aiohttp import web


async def proxy_handler(request: web.Request, target: str, logfile) -> web.StreamResponse:
    raw = await request.read()
    rec = {"ts": time.time(), "path": request.path,
           "h_body": hashlib.sha256(raw).hexdigest()[:16]}
    try:
        body = json.loads(raw)
        prompt = body.get("prompt")
        if isinstance(prompt, list):  # token ids 直投(本负载不会走这)
            prompt = json.dumps(prompt)
        if isinstance(prompt, str):
            rec["prompt_sha"] = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            rec["prompt_chars"] = len(prompt)
    except Exception:
        pass
    logfile.write(json.dumps(rec) + "\n")
    logfile.flush()

    async with aiohttp.ClientSession() as sess:
        async with sess.request(
            request.method, target + request.path_qs,
            data=raw, headers={k: v for k, v in request.headers.items()
                               if k.lower() not in ("host", "content-length")},
        ) as upstream:
            resp = web.StreamResponse(status=upstream.status,
                                      reason=upstream.reason)
            resp.headers.update({k: v for k, v in upstream.headers.items()
                                 if k.lower() not in
                                 ("content-length", "transfer-encoding",
                                  "connection")})
            await resp.prepare(request)
            async for chunk in upstream.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8901)
    ap.add_argument("--target", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    logfile = open(args.log, "w")

    async def handler(request: web.Request) -> web.StreamResponse:
        return await proxy_handler(request, args.target.rstrip("/"), logfile)

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{path:.*}", handler)
    print(f"[capture proxy] :{args.listen} -> {args.target} log={args.log}",
          flush=True)
    web.run_app(app, host="127.0.0.1", port=args.listen, print=None)


if __name__ == "__main__":
    main()
