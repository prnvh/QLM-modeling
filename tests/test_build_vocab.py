import json
import subprocess
import sys

import pytest

from lmf.infra.scripts.build_vocab import build_vocab_from_counts, preview_tokens


def test_build_vocab_cli_writes_metadata_and_human_readable_logs(tmp_path) -> None:
    input_path = tmp_path / "sample.txt"
    output_path = tmp_path / "vocab.json"
    log_path = tmp_path / "build.log"
    input_path.write_text("Bank bank.\nWithdraw money from bank!", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "lmf/infra/scripts/build_vocab.py",
            str(input_path),
            "--output",
            str(output_path),
            "--preview-tokens",
            "5",
            "--log-file",
            str(log_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    vocab = json.loads(output_path.read_text(encoding="utf-8"))
    terminal_log = result.stderr
    file_log = log_path.read_text(encoding="utf-8")

    assert result.stdout.startswith("Wrote ")
    assert vocab["metadata"]["num_files"] == 1
    assert vocab["metadata"]["total_tokens"] == 8
    assert vocab["token_to_id"]["<pad>"] == 0
    assert vocab["token_to_id"]["<unk>"] == 1
    assert vocab["token_counts"]["bank"] == 3
    assert "build_vocab.start" in terminal_log
    assert "build_vocab.tokenized_file" in terminal_log
    assert "preview=['bank', 'bank', '.', 'withdraw', 'money'" in terminal_log
    assert "build_vocab.vocabulary" in file_log
    assert "build_vocab.output" in file_log


@pytest.mark.parametrize(
    "tokens,limit,expected",
    [
        (["a", "b"], 0, "[... (+2)]"),
        (["a", "b"], 1, "['a', ... (+1)]"),
        (["a", "b"], 3, "['a', 'b']"),
    ],
)
def test_preview_tokens_is_human_readable(tokens, limit, expected) -> None:
    assert preview_tokens(tokens, limit=limit) == expected


def test_preview_tokens_rejects_negative_limit() -> None:
    with pytest.raises(ValueError):
        preview_tokens(["x"], limit=-1)


def test_build_vocab_from_counts_rejects_brittle_arguments() -> None:
    with pytest.raises(ValueError):
        build_vocab_from_counts({}, min_freq=0, max_size=None)

    with pytest.raises(ValueError):
        build_vocab_from_counts({}, min_freq=1, max_size=4)
