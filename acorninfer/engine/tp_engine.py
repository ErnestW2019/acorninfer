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
    """rank0 命令壳:submit/step/drop 先广播 (cmd, payload) 再本地执行。

    wire 只载**变更类**命令(submit/step/drop/fini)——pending/balanced 等
    查询 rank0 本地作答,不上广播线(SPMD 两 rank 状态同构,rank1 不需要
    知道查询结果;查询上线只会增加集合操作数,配对面毫无收益)。

    单飞锁(self._flight):任一时刻至多一个壳方法在飞行。server 的
    asyncio.Lock 已把 submit/step/drop 串行(锁跨 to_thread(step) 持有),
    但那是调用方纪律——壳自己保证集合操作 issue 顺序 = 程序序,不依赖
    任何调用方;否则两个线程并发进壳(如未来的健康检查直呼 has_pending)
    会让 rank0 的广播发射序不确定,而 rank1 单线程按己序消费,集合配对
    错位 → 死锁。

    已知项(M5T4 审查 Minor-1,本任务不改):rank1 对端死亡时 rank0 无快
    失败——卡在下一集合调用直到 NCCL watchdog(~600s)超时。有界可忍
    (正确性哨兵 check_views 的硬退路径不受影响;生产由 T6 PID kill 纪律
    兜底)。
    """

    def __init__(self, engine, link: TPLink):
        assert link.rank == 0, "TPEngine 壳只装在 rank0"
        self.engine = engine
        self.link = link
        self._flight = threading.Lock()

    def submit(self, prompt_ids, sampling, *, max_tokens, eos_ids=None):
        with self._flight:
            self.link.command(
                ("submit", (list(prompt_ids), sampling, max_tokens, eos_ids)))
            return self.engine.submit(prompt_ids, sampling,
                                      max_tokens=max_tokens, eos_ids=eos_ids)

    def step(self):
        with self._flight:
            self.link.command(("step", None))
            return self.engine.step()

    def drop(self, seq_id):
        with self._flight:
            self.link.command(("drop", seq_id))
            return self.engine.drop(seq_id)

    def has_pending(self):
        # 查询本地作答:不上广播线(见类 docstring wire 收窄)
        with self._flight:
            return self.engine.has_pending()

    def assert_balanced(self):
        with self._flight:
            self.engine.assert_balanced()

    def shutdown(self):
        """FINI:广播 rank1 优雅收尾(rank0 侧退出路径调用)。"""
        with self._flight:
            self.link.command(("fini", None))

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def seqs(self):
        return self.engine.seqs


def tp_rank1_loop(engine, link: TPLink) -> None:
    """rank1 命令循环(消费端):对称执行到 fini。wire 与 TPEngine 一一对应
    (仅变更类命令;查询不上线)。"""
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
