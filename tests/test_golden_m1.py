"""M1 acceptance: KV-cache path aligns with HF transformers.

- greedy 512: our first 256 generated ids are identical to HF's (passes;
  argmax margins hold throughout, cache path verified to token parity);
- sampling (temp 0.6 / top_p 0.95 / top_k 20): teacher-forced per-step
  logprob of OUR chosen token under HF's warpers, max abs diff < 0.4
  (2026-10-27 spec 修订值;原 0.1 为标定错误,证据链见下).

Evidence chain that set the threshold (all on this prompt):

- ours vs HF teacher-forced: max |dlp| = 0.198 (first test run), 0.158
  (seeded diag run); 42/48 steps exactly 0.0, mean 0.011 — the max lives
  in a few flat steps;
- OUR engine vs itself (cache-decode vs full-recompute): max |dlp| = 0.132;
- control: HF vs ITSELF (generate decode path vs teacher-forced forward,
  same warpers both sides): max |dlp| = 0.194 — the reference implementation
  trips 0.1 too, so 0.1 sat below the bf16 decode-vs-fullforward noise
  floor of any correct engine, not a defect of ours;
- mechanism: no top-p boundary flips (kept sets identical at worst steps).
  bf16 ulp at |logit| 25-31 is 0.125-0.25; temp 0.6 leaves only 2-5 tokens
  under top_p 0.95, the top token carries p≈0.6-0.9, so a 1-2 ulp shift on
  it moves log-sum-exp by ~0.1-0.3 and every kept token's lp with it;
- P(one sampled run staying under 0.1) ≈ 0: per-step trip mass reached 1.0.

0.4 = observed cross-path worst 0.198 + headroom over the HF-self control
0.194. Distribution-semantics bugs (wrong truncation, wrong warper order)
still blow this gate up, and the greedy gate above proves 256-token argmax
parity through the cache path; sampler math is pinned bit-exact to the
installed HF warpers in tests/test_sampler.py.

Run with CUDA_VISIBLE_DEVICES=1 (GPU 0 hosts an unrelated service);
sequential protocol: ours loads, runs, moves to CPU, then HF loads.
"""

import pytest
import torch
import transformers

from suiren.generate import generate_cached
from suiren.models.qwen3 import from_pretrained
from suiren.sampler import Sampler, SamplingParams
from suiren.tokenizer import Tokenizer

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"
PROMPT = "The first three prime numbers are"
GREEDY_TOKENS = 512
ACCEPT_TOKENS = 256
SAMPLE_TOKENS = 48
PARAMS = SamplingParams(temperature=0.6, top_p=0.95, top_k=20)


def _free(model) -> None:
    model.to("cpu")
    torch.cuda.empty_cache()


def _load_hf():
    return transformers.AutoModelForCausalLM.from_pretrained(
        QWEN3_8B,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()


@pytest.mark.slow
def test_greedy_512_first_256_match_hf():
    tok = Tokenizer(f"{QWEN3_8B}/tokenizer.json")
    prompt_ids = tok.encode(PROMPT)

    ours = from_pretrained(QWEN3_8B)
    ids, _ = generate_cached(ours, prompt_ids, max_new_tokens=GREEDY_TOKENS)
    ours_new = ids[len(prompt_ids) :]
    _free(ours)

    hf = _load_hf()
    try:
        hf_ids = transformers.AutoTokenizer.from_pretrained(QWEN3_8B)(
            PROMPT, return_tensors="pt", add_special_tokens=False
        ).input_ids.to("cuda:0")
        hf_out = hf.generate(hf_ids, do_sample=False, max_new_tokens=ACCEPT_TOKENS)
        hf_new = hf_out[0][hf_ids.shape[1] :].tolist()
        assert len(ours_new) >= ACCEPT_TOKENS, f"ours stopped early at {len(ours_new)}"
        assert ours_new[:ACCEPT_TOKENS] == hf_new[:ACCEPT_TOKENS], (
            f"first divergence within 256 tokens:\nhf  = {hf_new[:ACCEPT_TOKENS]}\n"
            f"ours= {ours_new[:ACCEPT_TOKENS]}"
        )
    finally:
        _free(hf)


@pytest.mark.slow
def test_sampling_logprob_within_tolerance_of_hf():
    tok = Tokenizer(f"{QWEN3_8B}/tokenizer.json")
    prompt_ids = tok.encode(PROMPT)

    ours = from_pretrained(QWEN3_8B)
    ids, step_lps = generate_cached(
        ours, prompt_ids, max_new_tokens=SAMPLE_TOKENS, sampler=Sampler(PARAMS)
    )
    assert len(step_lps) > 0
    # 注意:尾部 EOS 不剥掉 —— 教师强制的 last logprob 对应"选 EOS 那一步"的分布,
    # 剥掉会令最后一个 pos+1 越界;带 EOS 的序列照常前向即可。
    _free(ours)

    hf = _load_hf()
    try:
        full = torch.tensor([ids], device="cuda:0")
        with torch.inference_mode():
            hf_logits = hf(full).logits[0].float()  # [T, vocab]
        # transformers 5.12.1:独立的 _get_logits_warper 已并入 _get_logits_processor
        # (generation/utils.py:do_sample 分支,generate() 内部同款调用)。
        # 裸 sampling config 恰好返回 Temperature -> TopK -> TopP 三个 warper,
        # __call__(input_ids, scores) 签名不变;断言不动,仅适配构造。
        warpers = hf._get_logits_processor(
            transformers.GenerationConfig(
                temperature=PARAMS.temperature,
                top_p=PARAMS.top_p,
                top_k=PARAMS.top_k,
                do_sample=True,
            )
        )
        max_diff = 0.0
        for t, our_lp in enumerate(step_lps):
            pos = len(prompt_ids) - 1 + t  # 位置 pos 的分布预测 token pos+1
            scores = hf_logits[pos][None]
            for w in warpers:
                scores = w(full[:, : pos + 1], scores)
            hf_lp = float(torch.log_softmax(scores[0], -1)[ids[pos + 1]])
            max_diff = max(max_diff, abs(hf_lp - our_lp))
        # 0.4 = 2026-10-27 spec 修订值:bf16 跨前向路径噪声底实测(HF 自身对照 0.194,
        # 我们引擎内部 cache-vs-全量重算对照 0.132,本测试历次运行 0.158-0.198)。
        # 0.1 会把任何正确的 bf16 实现都判死;分布语义错误仍会被 max+greedy 门抓住。
        assert max_diff < 0.4, f"per-step logprob max abs diff = {max_diff}"
    finally:
        _free(hf)
