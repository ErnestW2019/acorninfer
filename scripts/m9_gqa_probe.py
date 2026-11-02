"""M9 Task 0 探针腿 a:SDPA enable_gqa 替代 _repeat_kv 的 A/B(临时探针,不进产线)。

跑法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m9_gqa_probe.py

对拍(逐位,非 TOL):同函数同输入下
  ref  = SDPA(q, _repeat_kv(k), _repeat_kv(v), is_causal/mask)   # 产线现路径
  gqa  = SDPA(q, k, v, enable_gqa=True, is_causal/mask)
torch.equal 逐位判定(自比较同输出 buffer,排除初始化噪声);另对拍
_repeat_kv 的 expand+reshape 物化拷贝与 enable_gqa 直读的数值等价性由 SDPA
内部语义保证 —— 本探针只钉"端到端输出逐位"。

计时:CUDA 事件对,50 次迭代取 mean/step,三条腿:
  A 产线同构(逐 b 循环 + repeat + SDPA)
  B 逐 b 循环 + enable_gqa(去物化拷贝)
  C 全批单调用 [N,H,L,D] + enable_gqa + is_causal(等长批才合法,
    bench 输入恰为等长 128 —— T1 1a 的目标形态,附测供参考)
另单测 _repeat_kv 本体拷贝耗时(cost model 用)。

形状:prefill 腿 L ∈ {1,15,16,17,128}(is_causal=(L>1),含 L=1 单 query
全可见特例);extend 腿 past ∈ {16,48} × L=128(bottom-right 显式 mask,
tril(diagonal=past))。Qwen3-8B: H=32, KV=8, rep=4, D=128, bf16。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

N, H, KV, D, REP = 8, 32, 8, 128, 4
ITERS = 50


def _repeat_kv(x: torch.Tensor, repeat: int) -> torch.Tensor:
    # qwen3.py:98 同款
    if repeat == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, repeat, s, d).reshape(b, h * repeat, s, d)


def _loop(q, k, v, *, use_gqa, starts=None):
    """产线 forward_prefill_batch b 循环同构(starts=None=prefill 腿,
    is_causal=(L>1);否则 extend 腿,bottom-right 显式 mask)。返回
    outs(逐 b 写入,同产线 outs 形态)。"""
    outs = torch.zeros(N, H, q.shape[2], D, dtype=q.dtype, device=q.device)
    for b in range(N):
        L = q.shape[2]
        if starts is None:
            if use_gqa:
                outs[b] = F.scaled_dot_product_attention(
                    q[b: b + 1], k[b: b + 1], v[b: b + 1],
                    is_causal=(L > 1), enable_gqa=True)[0]
            else:
                k_f = _repeat_kv(k[b: b + 1], REP)
                v_f = _repeat_kv(v[b: b + 1], REP)
                outs[b] = F.scaled_dot_product_attention(
                    q[b: b + 1], k_f, v_f, is_causal=(L > 1))[0]
        else:
            past = starts[b]
            mask = torch.ones(L, past + L, dtype=torch.bool,
                              device=q.device).tril(diagonal=past)[None, None]
            if use_gqa:
                outs[b] = F.scaled_dot_product_attention(
                    q[b: b + 1], k[b: b + 1], v[b: b + 1],
                    attn_mask=mask, enable_gqa=True)[0]
            else:
                k_f = _repeat_kv(k[b: b + 1], REP)
                v_f = _repeat_kv(v[b: b + 1], REP)
                outs[b] = F.scaled_dot_product_attention(
                    q[b: b + 1], k_f, v_f, attn_mask=mask)[0]
    return outs


def _batched(q, k, v):
    """腿 C:等长批单调用(enable_gqa + is_causal)。"""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          enable_gqa=True)


def _timeit(fn) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(ITERS):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / ITERS * 1000  # µs


def main() -> None:
    dev = "cuda:0"
    print(f"shapes: N={N} H={H} KV={KV} D={D} rep={REP} bf16 iters={ITERS}")
    gen = torch.Generator(device=dev)
    print("\n== 逐位对拍(同函数同输入,torch.equal;5 seeds × 形状)==")
    all_pass = True
    for seed in range(5):
        gen.manual_seed(20261027 + seed)
        q = torch.randn(N, H, 128, D, generator=gen, device=dev,
                        dtype=torch.bfloat16)
        k = torch.randn(N, KV, 128, D, generator=gen, device=dev,
                        dtype=torch.bfloat16)
        v = torch.randn(N, KV, 128, D, generator=gen, device=dev,
                        dtype=torch.bfloat16)
        for L in (1, 15, 16, 17, 128):
            ref = _loop(q[:, :, :L], k[:, :, :L], v[:, :, :L], use_gqa=False)
            gqa = _loop(q[:, :, :L], k[:, :, :L], v[:, :, :L], use_gqa=True)
            ok = torch.equal(ref, gqa)
            all_pass &= ok
            if not ok:
                d = (ref.view(torch.int16) - gqa.view(torch.int16)).abs().max()
                print(f"  seed={seed} L={L:<3} prefill  BITWISE FAIL maxdiff16={d}")
        for past in (16, 48):
            starts = [past] * N
            qe = q[:, :, :128]
            ke = torch.randn(N, KV, past + 128, D, generator=gen, device=dev,
                             dtype=torch.bfloat16)
            ve = torch.randn(N, KV, past + 128, D, generator=gen, device=dev,
                             dtype=torch.bfloat16)
            ref = _loop(qe, ke, ve, use_gqa=False, starts=starts)
            gqa = _loop(qe, ke, ve, use_gqa=True, starts=starts)
            ok = torch.equal(ref, gqa)
            all_pass &= ok
            if not ok:
                d = (ref.view(torch.int16) - gqa.view(torch.int16)).abs().max()
                print(f"  seed={seed} past={past} extend BITWISE FAIL maxdiff16={d}")
        # 腿 C 对拍:等长批单调用 vs 腿 A(prefill L=128)
        ref = _loop(q, k, v, use_gqa=False)
        bat = _batched(q, k, v)
        okc = torch.equal(ref, bat)
        all_pass &= okc
        if not okc:
            d = (ref.view(torch.int16) - bat.view(torch.int16)).abs().max()
            print(f"  seed={seed} L=128 batched-C BITWISE FAIL maxdiff16={d}")
    print(f"BITWISE-ALL {'PASS' if all_pass else 'FAIL'}"
          f"({'全部形状 × seeds 逐位相等' if all_pass else ',见上行 FAIL 明细'})")

    print("\n== 计时(mean µs/调用组)==")
    gen.manual_seed(20261027)
    q = torch.randn(N, H, 128, D, generator=gen, device=dev, dtype=torch.bfloat16)
    k = torch.randn(N, KV, 128, D, generator=gen, device=dev, dtype=torch.bfloat16)
    v = torch.randn(N, KV, 128, D, generator=gen, device=dev, dtype=torch.bfloat16)
    kb = torch.randn(N, KV, 48 + 128, D, generator=gen, device=dev,
                     dtype=torch.bfloat16)
    vb = torch.randn(N, KV, 48 + 128, D, generator=gen, device=dev,
                     dtype=torch.bfloat16)

    t_a = _timeit(lambda: _loop(q, k, v, use_gqa=False))
    t_b = _timeit(lambda: _loop(q, k, v, use_gqa=True))
    t_c = _timeit(lambda: _batched(q, k, v))
    t_ae = _timeit(lambda: _loop(q, kb, vb, use_gqa=False, starts=[48] * N))
    t_be = _timeit(lambda: _loop(q, kb, vb, use_gqa=True, starts=[48] * N))
    t_rep = _timeit(lambda: (_repeat_kv(k, REP), _repeat_kv(v, REP)))
    print(f"  prefill L=128 N=8:  A 产线(逐b+repeat+SDPA) {t_a:9.1f} µs")
    print(f"                      B 逐b+enable_gqa        {t_b:9.1f} µs")
    print(f"                      C 全批单调用+enable_gqa {t_c:9.1f} µs")
    print(f"  extend past=48 L=128 N=8: A {t_ae:9.1f} µs / B {t_be:9.1f} µs")
    print(f"  _repeat_kv 本体(k,v × 8 序列){t_rep:9.1f} µs")
    print(f"  腿A−腿B = {t_a - t_b:.1f} µs(每次前向每层;×36 层 = "
          f"{(t_a - t_b) * 36 / 1000:.2f} ms/prefill)")


if __name__ == "__main__":
    main()
