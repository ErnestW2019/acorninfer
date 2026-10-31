"""TP=2 SPMD 接线的单卡合同测试(M5 Task 4):不起 NCCL、不加载真模型。

覆盖:TPEngine 命令线 wire 格式钉子(submit/step/drop/pending/fini)、
_build_engine 的 --tp 2 分支(分片 TP 模型构造 + link 注入 + 壳包裹;
--tp 缺省走原路径)、孤儿防护(parent guard:父死子必死)。
真双卡 server 冒烟属 T5。
"""

import os
import time
import types

import pytest
import torch

from acorninfer.engine.tp_engine import TPEngine, install_parent_guard
from acorninfer.sampler import SamplingParams


class FakeLink:
    """rank0 侧假 link:command 记录 wire;sync/check 原样回(不起 dist)。"""

    def __init__(self, rank=0):
        self.rank = rank
        self.commands = []

    def command(self, cmd):
        self.commands.append(cmd)

    def sync_tokens(self, tokens):
        return tokens

    def check_views(self, views):
        pass


class FakeEngine:
    """记录调用的假 engine;返回值固定,供 passthrough 断言。"""

    def __init__(self):
        self.calls = []
        self.tokenizer = "TOK"
        self.seqs = {"s": "SEQ"}

    def submit(self, prompt_ids, sampling, *, max_tokens, eos_ids=None):
        self.calls.append(("submit", prompt_ids, sampling, max_tokens, eos_ids))
        return 42

    def step(self):
        self.calls.append(("step",))
        return ["OUT"]

    def drop(self, seq_id):
        self.calls.append(("drop", seq_id))

    def has_pending(self):
        self.calls.append(("pending",))
        return False

    def assert_balanced(self):
        self.calls.append(("balanced",))


def test_tp_engine_command_wire():
    """TPEngine 命令线:**只载变更类命令**(submit/step/drop/fini);
    pending/balanced 查询 rank0 本地作答不上广播线(wire 收窄,M5T4 审查);
    每个变更命令 = 一次广播 + 一次本地调用,返回值透传。"""
    eng = FakeEngine()
    link = FakeLink()
    shell = TPEngine(eng, link)

    sid = shell.submit([1, 2], SamplingParams(temperature=0.0), max_tokens=5)
    assert sid == 42  # 本地结果透传
    assert link.commands[-1] == ("submit", ([1, 2], SamplingParams(temperature=0.0), 5, None))
    assert eng.calls[-1] == ("submit", [1, 2], SamplingParams(temperature=0.0), 5, None)

    assert shell.step() == ["OUT"]
    assert link.commands[-1] == ("step", None)
    assert eng.calls[-1] == ("step",)

    shell.drop(42)
    assert link.commands[-1] == ("drop", 42)
    assert eng.calls[-1] == ("drop", 42)

    # 查询:本地作答,零广播(wire 收窄)
    assert shell.has_pending() is False
    assert eng.calls[-1] == ("pending",)
    shell.assert_balanced()
    assert eng.calls[-1] == ("balanced",)
    assert [c[0] for c in link.commands] == ["submit", "step", "drop"]  # 无 pending/balanced

    shell.shutdown()
    assert link.commands[-1] == ("fini", None)

    assert shell.tokenizer == "TOK"  # server 依赖的属性直通
    assert shell.seqs == {"s": "SEQ"}


def test_tp_engine_single_flight_serializes():
    """单飞锁:并发线程进壳,集合操作 issue 顺序 = 程序序(不依赖调用方
    纪律)。假 engine 记录 (进入, 退出) 事件,断言任意时刻至多一个调用在飞。"""
    import threading
    import time as _time

    class SlowEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.events = []

        def _fly(self, tag):
            self.events.append(("enter", tag))
            _time.sleep(0.02)
            self.events.append(("exit", tag))

        def submit(self, prompt_ids, sampling, *, max_tokens, eos_ids=None):
            self._fly("submit")
            return 1

        def step(self):
            self._fly("step")
            return []

    eng = SlowEngine()
    shell = TPEngine(eng, FakeLink())

    def worker(fn):
        for _ in range(4):
            fn()

    t1 = threading.Thread(target=worker, args=(shell.step,))
    t2 = threading.Thread(target=worker, args=(shell.step,))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert len(eng.events) == 16
    for i in range(0, 16, 2):
        assert eng.events[i][0] == "enter" and eng.events[i + 1][0] == "exit", (
            f"调用在飞行中被交叠:{eng.events[i:i + 2]}"
        )


def test_build_engine_tp2_uses_sharded_tp_model(monkeypatch):
    """_build_engine --tp 2:分片加载器构造 TP 模型(不走 from_pretrained)、
    attention 后端开关照旧置位、Engine 注入 tp link、外壳 TPEngine 包裹。"""
    import acorninfer.server as server_mod

    created: dict = {}

    def fake_sharded(model_dir, shard_fn, rank, world, *, device,
                     dtype=torch.bfloat16):
        created["sharded"] = (model_dir, rank, world, device, dtype)
        return {"w": "SHARDED"}

    def fake_tp_model(cfg, rank, world):
        created["tp_model"] = (cfg, rank, world)
        m = types.SimpleNamespace(cfg=cfg)

        def load(state, strict=True, assign=True):
            created["load_flags"] = (strict, assign)
        m.load_state_dict = load
        m.to = lambda device=None, dtype=None: m
        m.eval = lambda: m
        return m

    def fake_engine(model, tok, **kwargs):
        created["engine"] = (model, tok, kwargs)
        return "ENGINE"

    def fake_shell(engine, link):
        created["shell"] = (engine, link)
        return "TPENGINE"

    def fail_from_pretrained(*a, **k):
        raise AssertionError("--tp 2 不得走单卡 from_pretrained")

    monkeypatch.setattr(server_mod, "load_safetensors_sharded", fake_sharded)
    monkeypatch.setattr(server_mod, "Qwen3Config",
                        types.SimpleNamespace(from_json=lambda p: "CFG"))
    monkeypatch.setattr(server_mod, "build_shard_fn",
                        lambda cfg, rank, world: "SHARD_FN")
    monkeypatch.setattr(server_mod, "Qwen3TPForCausalLM", fake_tp_model)
    monkeypatch.setattr(server_mod, "from_pretrained", fail_from_pretrained)
    monkeypatch.setattr(server_mod, "Engine", fake_engine)
    monkeypatch.setattr(server_mod, "TPEngine", fake_shell)
    monkeypatch.setattr(server_mod, "Tokenizer", lambda path: "TOK")

    link = FakeLink(rank=0)
    args = types.SimpleNamespace(
        model_dir="/m", device="cuda:0", num_blocks=8, page_size=16,
        max_running=4, watermark_blocks=2, chunked_budget=None, tp=2,
    )
    assert server_mod._build_engine(args, link=link) == "TPENGINE"
    assert created["sharded"][:3] == ("/m", 0, 2)
    assert created["tp_model"][1:] == (0, 2)
    model, tok, kwargs = created["engine"]
    assert kwargs["tp"] is link
    assert kwargs["chunked_budget"] is None
    assert model.attention_backend == "sdpa"       # 后端开关与单卡构造同源
    assert model.decode_graph is False
    assert created["shell"] == ("ENGINE", link)


def test_build_engine_tp_default_is_legacy(monkeypatch):
    """--tp 缺省(1):逐字节原路径——from_pretrained 构造、Engine 不带 tp、
    不包壳;args 无 tp 属性(旧调用方)同样工作。"""
    import acorninfer.server as server_mod

    created: dict = {}

    def fake_from_pretrained(model_dir, device, dtype):
        created["fp"] = (model_dir, device, dtype)
        return types.SimpleNamespace()

    def fake_engine(model, tok, **kwargs):
        created["engine"] = kwargs
        return "ENGINE"

    def fail_shell(*a, **k):
        raise AssertionError("tp=1 不得包 TPEngine 壳")

    monkeypatch.setattr(server_mod, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(server_mod, "Engine", fake_engine)
    monkeypatch.setattr(server_mod, "TPEngine", fail_shell)
    monkeypatch.setattr(server_mod, "Tokenizer", lambda path: "TOK")

    args = types.SimpleNamespace(
        model_dir="/m", device="cuda:0", num_blocks=8, page_size=16,
        max_running=4, watermark_blocks=2, chunked_budget=None,
    )
    assert server_mod._build_engine(args) == "ENGINE"
    assert "tp" not in created["engine"]           # 缺省不注入 tp(硬门:现行为逐字节)


_GUARD_CHILD_SCRIPT = """
import os, sys, time
sys.path.insert(0, {root!r})
from acorninfer.engine.tp_engine import install_parent_guard

pid = os.fork()
if pid == 0:
    # C:装 guard(就绪文件先行),stdout/stderr 换 /dev/null(不拖住管道)
    try:
        install_parent_guard(poll_s=0.1)
    except Exception:
        os._exit(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    with open({ready!r}, "w") as f:
        f.write("ok")
    time.sleep(30)
    os._exit(0)
# P:等 C 装 guard 就绪,再退出 —— C 带 guard 成孤儿(PDEATHSIG 真实触发)
for _ in range(100):
    if os.path.exists({ready!r}):
        break
    time.sleep(0.05)
os.write(1, str(pid).encode())
"""


def test_tp_parent_guard_kills_orphan(tmp_path):
    """孤儿防护硬要求:P fork C(装 guard)后退出 → C 必死(GPU 必放)。

    中间父跑在独立 subprocess(干净单线程,fork 无死锁面);PDEATHSIG +
    轮询兜底双机制;死亡判据 = /proc 消失或僵尸(进程已终止,显存已放)。
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    ready = str(tmp_path / "guard_ready")
    script = _GUARD_CHILD_SCRIPT.format(root=root, ready=ready)
    p = subprocess.run([_sys.executable, "-c", script], capture_output=True,
                       timeout=60)
    assert p.returncode == 0, p.stderr.decode()[-400:]
    c_pid = int(p.stdout.decode().strip())
    assert (tmp_path / "guard_ready").exists(), "C 未报告 guard 就绪"

    deadline = time.time() + 5.0
    state = "?"
    while time.time() < deadline:
        try:
            with open(f"/proc/{c_pid}/stat") as f:
                state = f.read().split(") ", 1)[1][0]
        except FileNotFoundError:
            state = "GONE"
            break
        if state == "Z":
            break
        time.sleep(0.05)
    assert state in ("GONE", "Z"), (
        f"孤儿未被 guard 杀死(/proc/{c_pid} state={state!r})"
    )


# ---------- 真 server e2e(双卡;M5T4 审查补课)----------


def _make_serve_ckpt(ckpt_dir) -> None:
    """tiny ckpt:config.json + model.safetensors + tokenizer.json(拷 8B,
    e2e 用 token-ids prompt 不触发 encode;decode 对任意 id 合法)。"""
    import json as _json
    import shutil
    from pathlib import Path as _Path

    import torch as _torch
    from safetensors.torch import save_file

    from acorninfer.config import Qwen3Config as _Cfg
    from acorninfer.models.qwen3 import Qwen3ForCausalLM as _Full

    cfg = _Cfg.tiny()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(_json.dumps(dict(
        num_hidden_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        vocab_size=cfg.vocab_size,
        tie_word_embeddings=cfg.tie_word_embeddings,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=list(cfg.eos_token_id),
    )))
    _torch.manual_seed(0)
    full = _Full(cfg)
    sd = {k: v.contiguous() for k, v in full.state_dict().items()}
    save_file(sd, str(ckpt_dir / "model.safetensors"))
    tok_src = _Path("/home/ernest/model-weight/Qwen3-8B/tokenizer.json")
    if not tok_src.exists():
        pytest.skip("本机无 Qwen3-8B tokenizer.json,e2e 词汇表来源缺失")
    shutil.copy(tok_src, ckpt_dir / "tokenizer.json")


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="TP=2 需要 2 张卡")
def test_tp2_server_e2e_http(tmp_path):
    """真 uvicorn --tp 2 起服 + 真 HTTP 首条 /v1/completions(M5T4 审查
    Blocker 的结构性漏洞:此前无任何测试覆盖真 server 双卡路径——Blocker
    (rank1 模型被 cfg.device 放上 GPU0,comm/张量跨卡脑裂死锁)正是漏网)。

    断言面:health 就绪 → 首条 completions 200 + token 合理 → 复发请求
    仍 200 → 收服干净(T6 纪律:定向 kill、端口闭、双卡回落)。
    """
    import httpx
    import socket as _socket
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    ckpt = tmp_path / "serve_ckpt"
    _make_serve_ckpt(ckpt)

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
    log0 = (tmp_path / "rank0.log").open("w")   # rank1 继承同 stdio
    proc = _subprocess.Popen(
        [_sys.executable, "-m", "acorninfer.server",
         "--model-dir", str(ckpt), "--tp", "2", "--port", str(port)],
        cwd=str(root), env=env, stdout=log0, stderr=log0,
    )
    rank1_pid = None
    try:
        # health 轮询(就绪语境 = Blocker 复现条件的一部分)
        deadline = time.time() + 90
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(f"server 提前退出 rc={proc.returncode}")
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except (httpx.HTTPError, OSError):
                pass
            time.sleep(0.3)
        assert ready, "server 90s 未就绪"

        # 找 rank1(spawn 子进程)供收服断言
        out = _subprocess.run(["ps", "--ppid", str(proc.pid), "-o", "pid="],
                              capture_output=True, text=True).stdout
        for line in out.splitlines():
            cmd = _subprocess.run(
                ["ps", "-p", line.strip(), "-o", "cmd="],
                capture_output=True, text=True).stdout
            if "spawn_main" in cmd:
                rank1_pid = int(line.strip())
        assert rank1_pid is not None, "未找到 rank1 子进程"

        # 首条 completions(Blocker 场景)
        r = httpx.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json={"model": "tiny", "prompt": [7, 11, 13, 200, 45],
                  "max_tokens": 4, "temperature": 0},
            timeout=60,
        )
        assert r.status_code == 200, r.text[:400]
        body = r.json()
        choice = body["choices"][0]
        assert choice["finish_reason"] == "length"
        assert body["usage"]["completion_tokens"] == 4
        assert body["usage"]["prompt_tokens"] == 5

        # 复发请求(非仅首条)
        r2 = httpx.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json={"model": "tiny", "prompt": [123, 5, 77, 9, 21],
                  "max_tokens": 3, "temperature": 0},
            timeout=60,
        )
        assert r2.status_code == 200, r2.text[:400]
        assert r2.json()["usage"]["completion_tokens"] == 3
    finally:
        # T6 收服纪律:SIGTERM → 限期 → SIGKILL 定向;禁 pkill -f
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except _subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        if rank1_pid is not None:
            deadline = time.time() + 10
            while time.time() < deadline:
                if not _Path(f"/proc/{rank1_pid}").exists():
                    break
                time.sleep(0.2)
            else:
                os.kill(rank1_pid, 9)  # PDEATHSIG 失守 → 定向硬杀
        log0.close()

    # 端口闭 + 双卡回落(compute-apps 按 pid 断言,不受同机他人进程干扰)
    try:
        _socket.create_connection(("127.0.0.1", port), timeout=1)
        raise AssertionError(f"端口 {port} 在收服后仍开放")
    except OSError:
        pass
    out = _subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    live = {int(x) for x in out.split() if x.isdigit()}
    assert proc.pid not in live, "rank0 仍占 GPU"
    assert rank1_pid not in live, "rank1 仍占 GPU"
