import json
import subprocess
import sys

from lmf.core.input.tokenizer import build_vocabulary_from_texts


def test_inspect_tokenizer_pretty_report_controls_input_and_encoding() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/infra/scripts/inspect_tokenizer.py",
            "--text",
            "Help bank!",
            "--build-vocab-from-input",
            "--add-bos",
            "--add-eos",
            "--skip-special-decode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Input source: argument" in result.stdout
    assert "Input text:\nHelp bank!" in result.stdout
    assert '"help"' in result.stdout
    assert "Token IDs" in result.stdout
    assert "Decoded text:\nhelp bank!" in result.stdout


def test_inspect_tokenizer_json_report_shows_output_form(tmp_path) -> None:
    vocab = build_vocabulary_from_texts(["help bank withdraw"])
    vocab_path = tmp_path / "vocab.json"
    vocab.to_json(vocab_path)

    result = subprocess.run(
        [
            sys.executable,
            "lmf/infra/scripts/inspect_tokenizer.py",
            "--text",
            "Help money",
            "--vocab",
            str(vocab_path),
            "--add-bos",
            "--add-eos",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["input"]["text"] == "Help money"
    assert report["options"]["add_bos"] is True
    assert report["output"]["tokens"] == ["help", "money"]
    assert report["output"]["ids"][0] == vocab.bos_id
    assert report["output"]["ids"][2] == vocab.unk_id
    assert report["output"]["ids"][-1] == vocab.eos_id
    assert report["output"]["token_id_rows"][2]["token"] == "<unk>"


def test_inspect_tokenizer_writes_report_and_trace_log(tmp_path) -> None:
    input_path = tmp_path / "sample.txt"
    output_path = tmp_path / "report.json"
    log_path = tmp_path / "trace.log"
    input_path.write_text("Bank bank.", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "lmf/infra/scripts/inspect_tokenizer.py",
            "--input-file",
            str(input_path),
            "--build-vocab-from-input",
            "--format",
            "json",
            "--output",
            str(output_path),
            "--log-file",
            str(log_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    trace_log = log_path.read_text(encoding="utf-8")
    assert report["input"]["source"] == str(input_path)
    assert report["output"]["tokens"] == ["bank", "bank", "."]
    assert "tokenizer.tokenize" in trace_log
    assert "inspect_tokenizer.output" in trace_log
