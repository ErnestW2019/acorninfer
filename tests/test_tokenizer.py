from acorninfer.tokenizer import Tokenizer

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"


def make_tokenizer() -> Tokenizer:
    return Tokenizer(f"{QWEN3_8B}/tokenizer.json")


def test_roundtrip():
    tok = make_tokenizer()
    text = "The capital of France is"
    ids = tok.encode(text)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0
    assert tok.decode(ids) == text


def test_greedy_output_decodes():
    tok = make_tokenizer()
    ids = tok.encode("1 2 3 4 5")
    assert tok.decode(ids) == "1 2 3 4 5"
