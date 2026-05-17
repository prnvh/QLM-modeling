import logging
import subprocess
import sys

import pytest

from lmf.core.input.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    SimpleTokenizer,
    Vocabulary,
    build_vocabulary,
    build_vocabulary_from_texts,
    inspect_text,
    regex_tokenize,
)


def test_regex_tokenize_handles_common_edge_cases() -> None:
    text = "BANK, bank! can't re-enter cafe cafe, unicode café 😀"

    assert regex_tokenize("BANK bank") == ["bank", "bank"]
    assert regex_tokenize("BANK bank", lowercase=False) == ["BANK", "bank"]
    assert regex_tokenize(text) == [
        "bank",
        ",",
        "bank",
        "!",
        "can't",
        "re-enter",
        "cafe",
        "cafe",
        ",",
        "unicode",
        "café",
        "😀",
    ]


def test_build_vocabulary_has_stable_special_ids_and_ordering() -> None:
    vocab = build_vocabulary([["z", "a", "z", "b", "a", "c"]])

    assert vocab.id_to_token[:5] == [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1
    assert vocab.mask_id == 2
    assert vocab.bos_id == 3
    assert vocab.eos_id == 4
    assert vocab.id_to_token[5:] == ["a", "z", "b", "c"]


def test_encode_decode_unknown_padding_and_json_round_trip(tmp_path) -> None:
    vocab = build_vocabulary_from_texts(["Help me understand bank withdrawals. Help!"])
    tokenizer = SimpleTokenizer(vocab=vocab)

    ids = tokenizer.encode("Help unknown-token!", add_bos=True, add_eos=True, pad_to_length=8)

    assert ids == [vocab.bos_id, vocab.token_id("help"), vocab.unk_id, vocab.token_id("!"), vocab.eos_id, 0, 0, 0]
    assert tokenizer.decode(ids, skip_special=True) == "help!"

    vocab_path = tmp_path / "vocab.json"
    vocab.to_json(vocab_path)

    loaded = Vocabulary.from_json(vocab_path)
    assert loaded.token_to_id == vocab.token_to_id
    assert loaded.id_to_token == vocab.id_to_token


@pytest.mark.parametrize(
    "factory",
    [
        lambda: build_vocabulary([["x"]], min_freq=0),
        lambda: build_vocabulary([["x"]], max_size=4),
        lambda: Vocabulary({"<unk>": 1}),
        lambda: Vocabulary({"<unk>": 0, "x": 0}),
        lambda: Vocabulary.from_dict({"token_to_id": {"<unk>": 0}, "special_tokens": "<unk>"}),
        lambda: SimpleTokenizer(trace_limit=-1),
    ],
)
def test_invalid_inputs_fail_loudly(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_tokenizer_trace_logs_human_readable_steps(caplog) -> None:
    vocab = build_vocabulary_from_texts(["help bank"])
    tokenizer = SimpleTokenizer(vocab=vocab, trace=True, trace_limit=4)
    caplog.set_level(logging.INFO, logger="lmf.core.input.tokenizer")

    ids = tokenizer.encode("Help bank!", add_bos=True, add_eos=True)
    decoded = tokenizer.decode(ids, skip_special=True)

    messages = [record.message for record in caplog.records]
    assert decoded == "help bank"
    assert any("tokenizer.tokenize" in message and "tokens=['help', 'bank', '!']" in message for message in messages)
    assert any("tokenizer.encode.tokens" in message and "add_bos=True" in message for message in messages)
    assert any("tokenizer.encode.ids" in message and "id_count=5" in message for message in messages)
    assert any("tokenizer.decode" in message and "text=help bank" in message for message in messages)


def test_inspect_text_returns_controllable_tokenizer_packet() -> None:
    report = inspect_text("Help bank!")

    assert report["input"] == "Help bank!"
    assert report["tokens"] == ["help", "bank", "!"]
    assert report["ids"][0] == 3
    assert report["token_ids"][0] == (0, 3, "<bos>")
    assert report["decoded"] == "help bank!"


def test_tokenizer_file_can_be_run_directly() -> None:
    result = subprocess.run(
        [sys.executable, "lmf/core/input/tokenizer.py", "text", "Help bank!"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Input: Help bank!" in result.stdout
    assert "Tokens: ['help', 'bank', '!']" in result.stdout
    assert "Token ID rows:" in result.stdout
    assert "Decoded: help bank!" in result.stdout
