"""M9 Task 0 探针腿 c / 全链打点:chunked 拆块审计 + TTFT 链分段(临时探针,
不进产线;产线文件零改动 —— 全部经 monkeypatch,脚本留档复跑)。

跑法(CUDA_VISIBLE_DEVICES=1,8B bf16 满配方同构):
  # 1) tokenizer batch=8 encode 独立计时(bench 同款随机 128-token 语料)
  .venv/bin/python scripts/m9_chain_probe.py encode

  # 2) 引擎级 chunk/步审计:C=8 两种到达形态(burst / poisson λ=8),
  #    逐 step 打点 chunks 组合与 executor 耗时,逐请求账(默认 chunked_budget=None)
  .venv/bin/python scripts/m9_chain_probe.py engine-audit

  # 3) 打点 server(与 server.py main 的 tp=1 构造路径同构:同一 _build_engine
  #    + create_app;探针包装仅旁路读数。JSONL 写 --server-log)
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m9_chain_probe.py server \
      --port 8201 --attention-backend triton --fused-ops --cuda-graph

  # 4) 8 并发流式客户端(/v1/completions SSE,捕获 cmpl-{seq_id},与服务端
  #    JSONL 对齐成分段表):encode / 锁+排队 / 准入→首token / pump+flush+往返
  .venv/bin/python scripts/m9_chain_probe.py client --port 8201 \
      --server-log /tmp/opencode/m9t0/chain_server.jsonl

口径注记:
- 引擎单线程契约不变:探针只外层计时(schedule 返回值原样透传),不注入语义。
- executor 计时含采样同步(_sample 内 argmax→int 隐式同步)= 该步 GPU 全程
  (host 视角墙钟)。
- server 模式与 acorninfer.server 的差异仅两处:探针包装 + 独立 argparse
  (server.py 的 parser 在 main() 内联,禁改产线故此处按同默认值复制;
  逐项对照 server.py add_argument 清单)。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time

DEFAULT_MODEL = "/home/ernest/model-weight/Qwen3-8B"
DEFAULT_LOG = "/tmp/opencode/m9t0/chain_server.jsonl"
SEED = 20261027


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _hids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


# ------------------------------------------------------------ server 探针
class _Probe:
    """链路探针:实例级包装 Tokenizer.encode / Engine.submit /
    Scheduler.schedule / Executor.{prefill,prefill_batch,extend,decode_batch} /
    Engine.step,旁路记 JSONL。时间基准 = 服务端进程 time.perf_counter() − t0。"""

    def __init__(self, engine, path: str):
        self.engine = engine
        self.fh = open(path, "w")
        self.t0 = time.perf_counter()
        self._k = 0

    def w(self, **ev) -> None:
        ev["t"] = time.perf_counter() - self.t0
        self.fh.write(json.dumps(ev) + "\n")

    def install(self) -> None:
        eng = self.engine
        tok = eng.tokenizer
        orig_encode = tok.encode

        def encode(text: str) -> list[int]:
            t0 = time.perf_counter()
            ids = orig_encode(text)
            self.w(ev="encode", h_text=_h(text), h_ids=_hids(ids),
                   ms=(time.perf_counter() - t0) * 1e3)
            return ids

        tok.encode = encode

        orig_submit = eng.submit

        def submit(prompt_ids, sampling, **kw) -> int:
            seq_id = orig_submit(prompt_ids, sampling, **kw)
            self.w(ev="submit", seq_id=seq_id, h_ids=_hids(prompt_ids),
                   n=len(prompt_ids))
            return seq_id

        eng.submit = submit

        sched = eng.scheduler
        orig_schedule = sched.schedule

        def schedule():
            chunks, decodes = orig_schedule()
            self.w(ev="schedule", k=self._k,
                   chunks=[[s.seq_id, start, length]
                           for s, start, length in chunks],
                   decodes=[s.seq_id for s in decodes])
            return chunks, decodes

        sched.schedule = schedule

        exe = eng.executor
        for name in ("prefill", "prefill_batch", "extend", "decode_batch"):
            orig = getattr(exe, name)

            def mk(orig, name):
                def wrapped(*a, **kw):
                    t0 = time.perf_counter()
                    out = orig(*a, **kw)
                    self.w(ev="exec", k=self._k, kind=name,
                           ms=(time.perf_counter() - t0) * 1e3)
                    return out
                return wrapped

            setattr(exe, name, mk(orig, name))

        orig_step = eng.step

        def step():
            t0 = time.perf_counter()
            outs = orig_step()
            self.w(ev="step", k=self._k,
                   ms=(time.perf_counter() - t0) * 1e3,
                   outs=[[o.seq_id, o.token_id, o.finished] for o in outs])
            self._k += 1
            return outs

        eng.step = step


def _build_probe_argparser() -> argparse.ArgumentParser:
    """镜像 acorninfer.server main() 的 tp=1 参数面(默认值逐项对照;
    server parser 内联于 main(),产线禁改故此处复制)。"""
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=DEFAULT_MODEL)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-blocks", type=int, default=1024)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--max-running", type=int, default=8)
    p.add_argument("--watermark-blocks", type=int, default=8)
    p.add_argument("--chunked-budget", type=int, default=None)
    p.add_argument("--attention-backend", default="sdpa",
                   choices=["sdpa", "triton"])
    p.add_argument("--fused-ops", action="store_true")
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--quant", default="none", choices=["none", "w8a16", "w4a16"])
    p.add_argument("--spec-ngram", type=int, default=0)
    p.add_argument("--spec-num", type=int, default=4)
    p.add_argument("--server-log", default=None,
                   help="探针 JSONL 路径(探针专用,server.py 无此旗标)")
    return p


def mode_server(argv: list[str]) -> None:
    import uvicorn

    from acorninfer.server import _build_engine, create_app

    args = _build_probe_argparser().parse_args(argv)
    engine = _build_engine(args)
    log = args.server_log or DEFAULT_LOG
    probe = _Probe(engine, log)
    probe.install()
    print(f"[m9 chain probe] server on :{args.port}, log -> {log}",
          file=sys.stderr, flush=True)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


# -------------------------------------------------------------- client 模式
def mode_client(argv: list[str]) -> None:
    import aiohttp

    opts = {"port": 8201, "server_log": DEFAULT_LOG, "concurrency": 8,
            "out_tokens": 32}
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-").replace("-", "_")
        if key in opts:
            opts[key] = int(argv[i + 1]) if key != "server_log" else argv[i + 1]
            i += 2
        else:
            i += 1

    from acorninfer.bench import _prompts
    from acorninfer.tokenizer import Tokenizer

    tok = Tokenizer(f"{DEFAULT_MODEL}/tokenizer.json")
    vocab = json.load(open(f"{DEFAULT_MODEL}/config.json"))["vocab_size"]
    n = opts["concurrency"]
    ids = _prompts(n, 128, vocab, SEED)
    texts = [tok.decode(row) for row in ids]

    async def one(sess, idx, text):
        body = {"prompt": text, "max_tokens": opts["out_tokens"],
                "temperature": 0.0, "stream": True}
        t_send = time.perf_counter()
        seq_id = None
        first = None
        nch = 0
        buf = b""
        async with sess.post(
            f"http://127.0.0.1:{opts['port']}/v1/completions", json=body,
        ) as resp:
            async for raw in resp.content.iter_any():
                buf += raw
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    payload = json.loads(line[len(b"data: "):])
                    if seq_id is None:
                        seq_id = int(payload["id"].split("cmpl-")[1])
                        first = time.perf_counter()
                    nch += 1
        done = time.perf_counter()
        return {"idx": idx, "seq_id": seq_id, "h_text": _h(text),
                "ttft_ms": (first - t_send) * 1e3,
                "e2e_ms": (done - t_send) * 1e3, "chunks": nch}

    async def run():
        conn = aiohttp.TCPConnector(limit=n)
        async with aiohttp.ClientSession(
                connector=conn,
                timeout=aiohttp.ClientTimeout(total=600)) as sess:
            tasks = [asyncio.create_task(one(sess, i, t))
                     for i, t in enumerate(texts)]
            return await asyncio.gather(*tasks)

    rows = asyncio.run(run())
    print("CLIENT(per-request, submit 序):")
    for r in rows:
        print(f"  idx={r['idx']} seq={r['seq_id']} ttft={r['ttft_ms']:8.1f}ms "
              f"e2e={r['e2e_ms']:8.1f}ms chunks={r['chunks']} "
              f"h_text={r['h_text']}")
    ttfts = sorted(r["ttft_ms"] for r in rows)
    print(f"CLIENT TTFT ms: min={ttfts[0]:.1f} p50={ttfts[n // 2]:.1f} "
          f"max={ttfts[-1]:.1f}")

    if os.path.exists(opts["server_log"]):
        report_join(opts["server_log"], rows)


def report_join(server_log: str, rows) -> None:
    """server JSONL × client rows 对齐:分段全部用时长差值,跨时钟基只做
    ms 域减法(server 内段)与 ttft_c − server内总(= 上行+flush+往返)。"""
    enc, sub, sched, execs, steps = {}, {}, [], [], {}
    with open(server_log) as fh:
        for line in fh:
            ev = json.loads(line)
            t = ev["t"]
            if ev["ev"] == "encode":
                enc[ev["h_ids"]] = ev
            elif ev["ev"] == "submit":
                sub[ev["seq_id"]] = ev
            elif ev["ev"] == "schedule":
                sched.append(ev)
                for sid, _st, _ln in ev["chunks"]:
                    pass
            elif ev["ev"] == "exec":
                execs.append(ev)
            elif ev["ev"] == "step":
                steps[ev["k"]] = ev

    admit = {}   # seq_id -> (t, k) 首次出现在 chunks
    for ev in sched:
        for sid, _st, _ln in ev["chunks"]:
            admit.setdefault(sid, (ev["t"], ev["k"]))
    first_out = {}  # seq_id -> (k, t)
    for k, ev in steps.items():
        for sid, _tok, _fin in ev["outs"]:
            first_out.setdefault(sid, (k, ev["t"]))
    # 每步的 exec 时长(kind 取该步唯一 exec;prefill/extend/decode 一步一调用)
    exec_k = {ev["k"]: ev for ev in execs}

    print("\nJOIN(逐请求分段,ms;server 内段为服务端时基时长差):")
    print("  seq     ttft_client  encode  enc_end→submit  submit→admit  "
          "admit→首tok_step(步,k)  server内总(enc0→首tok)  ttft−server内总")
    out = []
    for r in rows:
        sid = r["seq_id"]
        s_ev = sub.get(sid)
        e = enc.get(s_ev["h_ids"]) if s_ev else None
        if e is None or sid not in admit:
            print(f"  seq={sid}: 对齐缺失(encode/submit/admit 不全)")
            continue
        t_sub = s_ev["t"]
        enc_ms = e["ms"]
        q_ms = (t_sub - (e["t"] + enc_ms / 1e3)) * 1e3
        a_t, a_k = admit[sid]
        a_ms = (a_t - t_sub) * 1e3
        k, st_t = first_out.get(sid, (None, float("nan")))
        f_ms = (st_t - a_t) * 1e3
        ek = exec_k.get(k, {})
        srv_total = (st_t - e["t"]) * 1e3
        post = r["ttft_ms"] - srv_total
        out.append((r["ttft_ms"], sid, enc_ms, q_ms, a_ms, f_ms, k,
                    ek.get("kind"), ek.get("ms"), srv_total, post))
    out.sort()
    for (ttft, sid, enc_ms, q_ms, a_ms, f_ms, k, kind, ems, srv, post) in out:
        print(f"  seq{sid:<4} {ttft:9.1f} {enc_ms:7.2f} {q_ms:10.2f} "
              f"{a_ms:10.2f}  {f_ms:9.1f} (k={k},{kind}={ems:.1f}ms) "
              f"{srv:9.1f} {post:9.1f}")
    print("口径:ttft_client = HTTP 发送→首 SSE 数据块;a→首tok_step 含该步"
          "全部排队/前向成员分摊(整批共享,见 exec ms);ttft−server内总 = "
          "上行+pydantic+pump 路由+SSE flush+网络往返。")


# --------------------------------------------------------- encode 模式
def mode_encode() -> None:
    from acorninfer.bench import _prompts
    from acorninfer.tokenizer import Tokenizer

    tok = Tokenizer(f"{DEFAULT_MODEL}/tokenizer.json")
    vocab = json.load(open(f"{DEFAULT_MODEL}/config.json"))["vocab_size"]
    ids = _prompts(8, 128, vocab, SEED)
    texts = [tok.decode(row) for row in ids]
    tok.encode(texts[0])  # 预热
    reps = 20
    totals = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for t in texts:
            tok.encode(t)
        totals.append((time.perf_counter() - t0) * 1e3)
    totals.sort()
    print(f"ENCODE batch=8(in=128 tok,文本复自 decode): 8 条逐条 encode 合计 "
          f"mean={sum(totals)/reps:.3f}ms p50={totals[reps//2]:.3f}ms "
          f"min={totals[0]:.3f}ms max={totals[-1]:.3f}ms")


# ---------------------------------------------------- engine-audit 模式
class _Audit:
    """engine-audit 轻量包装(内存态;同一 engine 串行跑两到达形态)。"""

    def __init__(self, engine):
        self.engine = engine
        self.reset()
        sched = engine.scheduler
        if not getattr(sched, "_m9_wrapped", False):
            orig_schedule = sched.schedule

            def schedule():
                chunks, decodes = orig_schedule()
                self._log_schedule(chunks, decodes)
                return chunks, decodes

            sched.schedule = schedule
            sched._m9_wrapped = True
        exe = engine.executor
        if not getattr(exe, "_m9_wrapped", False):
            for name in ("prefill", "prefill_batch", "extend", "decode_batch"):
                orig = getattr(exe, name)

                def mk(orig, name):
                    def wrapped(*a, **kw):
                        t0 = time.perf_counter()
                        out = orig(*a, **kw)
                        self._exec = (name, (time.perf_counter() - t0) * 1e3)
                        return out
                    return wrapped

                setattr(exe, name, mk(orig, name))
            exe._m9_wrapped = True
        orig_step = engine.step
        if not getattr(engine, "_m9_wrapped", False):
            def step():
                t0 = time.perf_counter()
                outs = orig_step()
                self._log_step(outs, (time.perf_counter() - t0) * 1e3)
                return outs
            engine.step = step
            engine._m9_wrapped = True

    def reset(self) -> None:
        self.t0 = time.perf_counter()
        self.lines: list[str] = []
        self._sub: dict[int, float] = {}
        self._adm: dict[int, float] = {}
        self._nchunk: dict[int, int] = {}
        self._first: dict[int, float] = {}
        self._done: dict[int, float] = {}
        self._k = 0
        self._exec = ("-", 0.0)
        self._pend_chunks: list[tuple[int, int, int]] = []
        self._pend_ndec = 0

    def _log_schedule(self, chunks, decodes):
        self._exec = ("-", 0.0)
        self._pend_chunks = [(s.seq_id, start, length)
                             for s, start, length in chunks]
        self._pend_ndec = len(decodes)
        now = time.perf_counter() - self.t0
        for s, _start, _ln in chunks:
            self._nchunk[s.seq_id] = self._nchunk.get(s.seq_id, 0) + 1
            if s.seq_id not in self._adm:
                self._adm[s.seq_id] = now

    def _log_step(self, outs, ms):
        now = time.perf_counter() - self.t0
        self.lines.append(
            f"step{self._k:3d}: t={now:7.3f}s wall={ms:7.2f}ms "
            f"exec({self._exec[0]})={self._exec[1]:7.2f}ms "
            f"chunks={self._pend_chunks} ndecodes={self._pend_ndec}")
        for o in outs:
            if o.seq_id not in self._first:
                self._first[o.seq_id] = now
            if o.finished:
                self._done[o.seq_id] = now
        self._k += 1

    def mark_submit(self, sid: int, t: float) -> None:
        self._sub[sid] = t


def mode_engine_audit() -> None:
    import torch

    from acorninfer.bench import _prompts
    from acorninfer.engine.engine import Engine
    from acorninfer.models.qwen3 import from_pretrained
    from acorninfer.sampler import SamplingParams
    from acorninfer.tokenizer import Tokenizer

    device = "cuda:0"
    model = from_pretrained(DEFAULT_MODEL, device, torch.bfloat16)
    model.attention_backend = "triton"
    model.fused_ops = True
    model.decode_graph = True
    tok = Tokenizer(f"{DEFAULT_MODEL}/tokenizer.json")
    vocab = model.cfg.vocab_size
    engine = Engine(model, tok, num_blocks=1024, page_size=16, max_running=8,
                    watermark_blocks=8, chunked_budget=None)
    aud = _Audit(engine)
    ids = _prompts(8, 128, vocab, SEED)

    g = torch.Generator().manual_seed(SEED)
    patterns = {
        "burst": [0.0] * 8,
        "poisson8": None,  # λ=8:指数间隔 mean 125ms
    }
    gaps = torch.empty(8).exponential_(lambd=8.0, generator=g).tolist()
    acc = 0.0
    poisson = []
    for gp in gaps:
        acc += gp
        poisson.append(acc)

    for pattern, arrivals in patterns.items():
        if arrivals is None:
            arrivals = poisson
        aud.reset()
        submitted = 0
        sids = []
        while submitted < 8 or engine.has_pending():
            now = time.perf_counter() - aud.t0
            while submitted < 8 and now >= arrivals[submitted]:
                sid = engine.submit(list(ids[submitted]),
                                    SamplingParams(temperature=0.0),
                                    max_tokens=32)
                aud._sub[sid] = time.perf_counter() - aud.t0
                sids.append(sid)
                submitted += 1
            if engine.has_pending():
                engine.step()
            elif submitted < 8:
                # 队空且尚有未到达请求:睡到下一到达点(不空转)
                time.sleep(max(0.0, arrivals[submitted]
                               - (time.perf_counter() - aud.t0)))
        print(f"\n== pattern={pattern}(λ=8 即 mean 间隔 125ms)==")
        print("step 负载账:")
        for line in aud.lines:
            print("  " + line)
        print("请求账:")
        for sid in sids:
            sub = aud._sub[sid]
            adm = aud._adm.get(sid, float("nan"))
            ft = aud._first.get(sid, float("nan"))
            dn = aud._done.get(sid, float("nan"))
            print(f"  seq{sid}: submit={sub:7.3f}s admit={adm:7.3f}s "
                  f"first_tok={ft:7.3f}s TTFT_server={(ft - sub) * 1e3:7.1f}ms "
                  f"chunks={aud._nchunk.get(sid, 0)} done={dn:7.3f}s "
                  f"(排队 {(adm - sub) * 1e3:6.1f}ms + 首 token "
                  f"{(ft - adm) * 1e3:6.1f}ms)")
    try:
        model.to("cpu")
    except Exception:
        pass
    torch.cuda.empty_cache()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if mode == "encode":
        mode_encode()
    elif mode == "engine-audit":
        mode_engine_audit()
    elif mode == "server":
        mode_server(rest)
        return
    elif mode == "client":
        mode_client(rest)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
