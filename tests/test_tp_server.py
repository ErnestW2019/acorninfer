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
    """TPEngine 命令线:每个公开入口先广播 (cmd, payload) 再本地执行;
    wire 与 tp_rank1_loop 的消费端一一对应(两端合同钉死)。"""
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

    assert shell.has_pending() is False
    assert link.commands[-1] == ("pending", None)

    shell.assert_balanced()
    assert link.commands[-1] == ("balanced", None)

    shell.shutdown()
    assert link.commands[-1] == ("fini", None)

    assert shell.tokenizer == "TOK"  # server 依赖的属性直通
    assert shell.seqs == {"s": "SEQ"}


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
