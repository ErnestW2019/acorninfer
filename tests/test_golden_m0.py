"""M0 acceptance: our greedy tokens match HF transformers token-for-token.

Spec: on fixed prompts, the first >= 50 generated tokens are identical.
Single-GPU sequential protocol (GPU 0 hosts an unrelated service):
run with CUDA_VISIBLE_DEVICES=1; ours generates and frees, then HF loads.

Prompt selection: the original draft used "The capital of France is" and
"Water boils at a temperature of", but those greedy trajectories cross
exact bf16 argmax ties within 50 tokens (e.g. France prompt, step 12:
fp32 true top-2 margin is 0.075 while cross-engine bf16 noise is ~0.2, so
both engines quantize " Spain"/" Germany" to the same logit and the token
is decided by tie-break order, not numerics). They are replaced by
same-flavor factual prompts verified divergence-free over all 60 tokens
(ours == HF exactly, strictly positive top-2 logit margins throughout).
"""

import pytest
import torch
import transformers

from suiren.config import Qwen3Config
from suiren.generate import generate_token_ids
from suiren.models.qwen3 import Qwen3ForCausalLM, map_hf_state_dict
from suiren.tokenizer import Tokenizer
from suiren.weights import load_safetensors

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"
PROMPTS = [
    "The first three prime numbers are",
    "The chemical symbol for gold is",
    "The Pacific Ocean is the largest",
]
ACCEPT_TOKENS = 50
GEN_TOKENS = ACCEPT_TOKENS + 10


def _free(model) -> None:
    # `del` in a helper only drops the local name, so the caller's reference
    # would keep the weights on GPU; move to CPU to release them for HF.
    model.to(device="cpu")
    torch.cuda.empty_cache()


def load_suiren() -> Qwen3ForCausalLM:
    cfg = Qwen3Config.from_json(f"{QWEN3_8B}/config.json")
    model = Qwen3ForCausalLM(cfg)  # CPU 构建,避免双份显存峰值
    mapped = map_hf_state_dict(load_safetensors(QWEN3_8B, device="cpu"))
    model.load_state_dict(mapped, strict=True, assign=True)
    return model.to(device="cuda:0", dtype=torch.bfloat16).eval()


def load_hf() -> transformers.PreTrainedModel:
    return transformers.AutoModelForCausalLM.from_pretrained(
        QWEN3_8B,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()


@pytest.mark.slow
def test_golden_alignment():
    suiren_tok = Tokenizer(f"{QWEN3_8B}/tokenizer.json")
    hf_tok = transformers.AutoTokenizer.from_pretrained(QWEN3_8B)

    # 第一阶段:我们的引擎生成(全量重算 oracle)
    ours_by_prompt = {}
    suiren_model = load_suiren()
    for prompt in PROMPTS:
        ids = generate_token_ids(
            suiren_model,
            tokenizer=suiren_tok,
            prompt=prompt,
            max_new_tokens=GEN_TOKENS,
        )
        ours_by_prompt[prompt] = ids[len(suiren_tok.encode(prompt)):]
    _free(suiren_model)

    # 第二阶段:HF 复现(显式关闭采样与 KV cache,与 oracle 路径对齐)
    hf_model = load_hf()
    try:
        # 逐 prompt 独立判定,一个失败不掩盖其余 prompt 的结果
        failures: list[str] = []
        for prompt in PROMPTS:
            hf_ids = hf_tok(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to("cuda:0")
            hf_out = hf_model.generate(
                hf_ids,
                do_sample=False,
                use_cache=False,
                max_new_tokens=GEN_TOKENS,
            )
            hf_new = hf_out[0][hf_ids.shape[1]:].tolist()
            ours_new = ours_by_prompt[prompt]

            common = min(len(hf_new), len(ours_new))
            try:
                assert common >= ACCEPT_TOKENS, (
                    f"prompt={prompt!r} generated only {common} common tokens"
                    f" (< {ACCEPT_TOKENS});\n"
                    f"hf  = {hf_new[:60]}\nours= {ours_new[:60]}"
                )
                assert hf_new[:ACCEPT_TOKENS] == ours_new[:ACCEPT_TOKENS], (
                    f"prompt={prompt!r} first {ACCEPT_TOKENS} tokens diverge;\n"
                    f"hf  = {hf_new[:ACCEPT_TOKENS]}\nours= {ours_new[:ACCEPT_TOKENS]}"
                )
            except AssertionError as e:
                failures.append(str(e))
        assert not failures, "\n\n".join(failures)
    finally:
        _free(hf_model)
