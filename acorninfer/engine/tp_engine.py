"""TP=2 SPMD 接线(M5 Task 4):命令级壳 + token 级同步 + 页表漂移哨兵 + 孤儿防护。

架构(spec 决策,时间场审计后落定):
- 命令级:TPEngine(rank0 壳)每个公开入口先广播 (cmd, payload) 再本地执行;
  rank1 跑 tp_rank1_loop 对称执行。scheduler 决策输入全同(FCFS 列表序、
  块表、allocator 状态;无 wall-time 分支——审计:created_at/first_token_at
  仅进 bench 指标,抢占"最老"= running.pop(0))→ 决策全同。
- token 级:采样点在 rank0(_sample / 合批 argmax),token ids 经
  TPLink.sync_tokens 广播,两 rank 同 token 进 _record;broadcast 在图外、
  采样点处。greedy decode 的 logits 经模型内 all-gather 两 rank 逐位同,
  广播对 greedy 是冗余保险;temp>0 的 RNG 路径(rank1 不碰采样器)则必须。
- 哨兵(R3):每步 check_views 广播 rank0 页表快照,rank1 逐位比对,漂移即
  硬退(os._exit——SPMD 不变式破坏后进程状态不可信,显存随进程消亡释放;
  rank0 在下一个集合调用快速失败)。频率取每步:快照为活跃视图的小表,
  pickle 开销远小于一步前向。
- 孤儿防护:rank1 install_parent_guard —— prctl PDEATHSIG(SIGKILL)+
  轮询兜底;rank0 死则 rank1 必死、GPU 必放(T6 PID kill 纪律依赖,不许
  新增孤儿类)。
"""

from __future__ import annotations

import os
import signal
import threading
import time

from acorninfer.distributed import broadcast_object


class TPLink:
    """token 级同步 + 哨兵句柄(两 rank 各持一个,rank 号区分角色)。"""

    def __init__(self, rank: int):
        self.rank = rank

    # ---- 命令级(rank0 壳 → rank1 循环)----

    def command(self, cmd):
        """rank0 传命令元组;rank1 传 None,返回收到的 (cmd, payload)。"""
        return broadcast_object(cmd if self.rank == 0 else None, src=0)

    # ---- 采样点同步(rank0 采样 → 两 rank 同 token)----

    def sync_tokens(self, tokens):
        """rank0 传采样结果(list[int|None]),原样返回;rank1 传 None,返回收到的。"""
        return broadcast_object(tokens if self.rank == 0 else None, src=0)

    # ---- 决策漂移哨兵(R3)----

    def check_views(self, views: dict) -> None:
        """rank0 广播页表快照,rank1 逐位比对;漂移 = SPMD 不变式破坏 → 硬退。"""
        snap = {
            sid: (list(v.table), [int(v._lens[k]) for k in sorted(v._lens)])
            for sid, v in views.items()
        }
        if self.rank == 0:
            broadcast_object(snap, src=0)
            return
        snap0 = broadcast_object(None, src=0)
        if snap0 != snap:
            drifted = sorted(
                {sid for sid in set(snap0) | set(snap)
                 if snap0.get(sid) != snap.get(sid)}
            )
            print(f"[TP-SENTINEL] 页表漂移(seq={drifted});rank1 硬退",
                  flush=True)
            os._exit(1)


class TPEngine:
    """rank0 命令壳:每个公开入口先广播 (cmd, payload) 再本地执行。

    wire 与 tp_rank1_loop 一一对应(submit/step/drop/pending/balanced/fini);
    server 所需属性(tokenizer/seqs)直通。
    """

    def __init__(self, engine, link: TPLink):
        assert link.rank == 0, "TPEngine 壳只装在 rank0"
        self.engine = engine
        self.link = link

    def submit(self, prompt_ids, sampling, *, max_tokens, eos_ids=None):
        self.link.command(
            ("submit", (list(prompt_ids), sampling, max_tokens, eos_ids)))
        return self.engine.submit(prompt_ids, sampling, max_tokens=max_tokens,
                                  eos_ids=eos_ids)

    def step(self):
        self.link.command(("step", None))
        return self.engine.step()

    def drop(self, seq_id):
        self.link.command(("drop", seq_id))
        return self.engine.drop(seq_id)

    def has_pending(self):
        self.link.command(("pending", None))
        return self.engine.has_pending()

    def assert_balanced(self):
        self.link.command(("balanced", None))
        self.engine.assert_balanced()

    def shutdown(self):
        """FINI:广播 rank1 优雅收尾(rank0 侧退出路径调用)。"""
        self.link.command(("fini", None))

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def seqs(self):
        return self.engine.seqs


def tp_rank1_loop(engine, link: TPLink) -> None:
    """rank1 命令循环(消费端):对称执行到 fini。"""
    while True:
        cmd, payload = link.command(None)
        if cmd == "submit":
            prompt_ids, sampling, max_tokens, eos_ids = payload
            engine.submit(prompt_ids, sampling, max_tokens=max_tokens,
                          eos_ids=eos_ids)
        elif cmd == "step":
            engine.step()
        elif cmd == "drop":
            engine.drop(payload)
        elif cmd == "pending":
            engine.has_pending()
        elif cmd == "balanced":
            engine.assert_balanced()
        elif cmd == "fini":
            return
        else:
            raise ValueError(f"unknown TP command {cmd!r}")


def install_parent_guard(poll_s: float = 5.0) -> None:
    """孤儿防护:prctl PDEATHSIG(SIGKILL;父线程死即杀)+ 轮询兜底。

    PDEATHSIG 不追溯(装护栏时父已死则不触发),故先查当前 getppid;
    轮询以"父 pid 变化"为判据(重父化即孤儿)。
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # PR_SET_PDEATHSIG = 1:父线程死 → 内核直接 SIGKILL 本进程
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno())
    except Exception:
        pass  # 平台不支持 prctl:轮询兜底仍在
    ppid = os.getppid()
    if ppid == 1:
        os._exit(1)  # 装护栏时父已死
    def _poll():
        while True:
            time.sleep(poll_s)
            if os.getppid() != ppid:
                os._exit(1)
    threading.Thread(target=_poll, daemon=True).start()
