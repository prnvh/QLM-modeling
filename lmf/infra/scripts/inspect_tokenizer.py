"""Inspect tokenizer input, output tokens, ids, and decoded text."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.input.tokenizer import (  # noqa: E402
    DEFAULT_TOKEN_PATTERN,
    SimpleTokenizer,
    Vocabulary,
    build_vocabulary_from_texts,
)

LOGGER = logging.getLogger("lmf.inspect_tokenizer")


def configure_logging(*, trace: bool, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = []
    if trace:
        handlers.append(logging.StreamHandler())
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    if not handlers:
        return

    formatter = logging.Formatter("%(levelname)s %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def read_input_text(args: argparse.Namespace) -> tuple[str, str]:
    provided = [args.text is not None, args.input_file is not None, args.stdin]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of --text, --input-file, or --stdin.")

    if args.text is not None:
        return args.text, "argument"
    if args.input_file is not None:
        return args.input_file.read_text(encoding=args.encoding), str(args.input_file)
    return sys.stdin.read(), "stdin"


def load_or_build_vocab(args: argparse.Namespace, text: str, *, lowercase: bool) -> tuple[Vocabulary | None, str]:
    if args.vocab is not None and args.build_vocab_from_input:
        raise ValueError("Use either --vocab or --build-vocab-from-input, not both.")
    if args.vocab is not None:
        return Vocabulary.from_json(args.vocab), str(args.vocab)
    if args.build_vocab_from_input:
        vocab = build_vocabulary_from_texts(
            [text],
            lowercase=lowercase,
            min_freq=args.min_freq,
            max_size=args.max_size,
            pattern=args.pattern,
        )
        return vocab, "built from current input"
    return None, "none"


def inspect_tokenizer(args: argparse.Namespace) -> dict[str, Any]:
    if args.trace_limit < 0:
        raise ValueError("--trace-limit must be non-negative.")
    if args.max_length is not None and args.max_length < 0:
        raise ValueError("--max-length must be non-negative.")
    if args.pad_to_length is not None and args.pad_to_length < 0:
        raise ValueError("--pad-to-length must be non-negative.")

    lowercase = not args.case_sensitive
    input_text, input_source = read_input_text(args)
    vocab, vocab_source = load_or_build_vocab(args, input_text, lowercase=lowercase)
    tokenizer = SimpleTokenizer(
        vocab=vocab,
        lowercase=lowercase,
        pattern=args.pattern,
        trace=args.trace or args.log_file is not None,
        trace_limit=args.trace_limit,
    )

    tokens = tokenizer.tokenize(input_text)
    ids: list[int] | None = None
    decoded: str | None = None
    token_id_rows: list[dict[str, int | str]] | None = None
    if vocab is not None:
        ids = tokenizer.encode(
            input_text,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
            max_length=args.max_length,
            pad_to_length=args.pad_to_length,
        )
        decoded = tokenizer.decode(ids, skip_special=args.skip_special_decode)
        token_id_rows = [
            {"position": idx, "id": token_id, "token": vocab.token(token_id)}
            for idx, token_id in enumerate(ids)
        ]

    return {
        "input": {
            "source": input_source,
            "text": input_text,
            "char_count": len(input_text),
        },
        "options": {
            "lowercase": lowercase,
            "pattern": args.pattern,
            "add_bos": args.add_bos,
            "add_eos": args.add_eos,
            "max_length": args.max_length,
            "pad_to_length": args.pad_to_length,
            "skip_special_decode": args.skip_special_decode,
        },
        "vocab": {
            "source": vocab_source,
            "size": len(vocab) if vocab is not None else None,
        },
        "output": {
            "tokens": tokens,
            "token_count": len(tokens),
            "ids": ids,
            "id_count": len(ids) if ids is not None else None,
            "token_id_rows": token_id_rows,
            "decoded": decoded,
        },
    }


def format_pretty(report: dict[str, Any]) -> str:
    output = report["output"]
    vocab = report["vocab"]
    lines = [
        "Tokenizer Inspection",
        "",
        f"Input source: {report['input']['source']}",
        f"Input chars: {report['input']['char_count']}",
        "Input text:",
        report["input"]["text"],
        "",
        "Options:",
        json.dumps(report["options"], indent=2),
        "",
        f"Vocab source: {vocab['source']}",
        f"Vocab size: {vocab['size']}",
        "",
        f"Tokens ({output['token_count']}):",
        json.dumps(output["tokens"], indent=2),
    ]

    if output["ids"] is None:
        lines.extend(
            [
                "",
                "Token IDs: not produced because no vocabulary was provided or built.",
                "Use --vocab PATH or --build-vocab-from-input to inspect ids and decoded text.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "",
            f"Token IDs ({output['id_count']}):",
            json.dumps(output["ids"], indent=2),
            "",
            "Token ID rows:",
        ]
    )
    for row in output["token_id_rows"]:
        lines.append(f"  {row['position']:>3} | {row['id']:>5} | {row['token']}")
    lines.extend(["", "Decoded text:", output["decoded"]])
    return "\n".join(lines)


def write_report(report: dict[str, Any], *, output_format: str, output_path: Path | None) -> None:
    if output_format == "json":
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    else:
        text = format_pretty(report) + "\n"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Literal input text to inspect.")
    source.add_argument("--input-file", type=Path, help="Text file to inspect.")
    source.add_argument("--stdin", action="store_true", help="Read input text from stdin.")

    vocab = parser.add_mutually_exclusive_group()
    vocab.add_argument("--vocab", type=Path, help="Vocabulary JSON to use for ids/decode.")
    vocab.add_argument(
        "--build-vocab-from-input",
        action="store_true",
        help="Build a temporary vocabulary from the provided input text.",
    )

    parser.add_argument("--case-sensitive", action="store_true", help="Keep original casing.")
    parser.add_argument("--pattern", default=DEFAULT_TOKEN_PATTERN, help="Regex token pattern.")
    parser.add_argument("--encoding", default="utf-8", help="Input file encoding.")
    parser.add_argument("--min-freq", type=int, default=1, help="Temporary vocab minimum frequency.")
    parser.add_argument("--max-size", type=int, default=None, help="Temporary vocab maximum size.")
    parser.add_argument("--add-bos", action="store_true", help="Add <bos> before encoding.")
    parser.add_argument("--add-eos", action="store_true", help="Add <eos> after encoding.")
    parser.add_argument("--max-length", type=int, default=None, help="Truncate encoded ids to this length.")
    parser.add_argument("--pad-to-length", type=int, default=None, help="Pad/truncate encoded ids to this length.")
    parser.add_argument(
        "--skip-special-decode",
        action="store_true",
        help="Hide special tokens when showing decoded text.",
    )
    parser.add_argument("--trace", action="store_true", help="Print tokenizer trace logs to stderr.")
    parser.add_argument("--trace-limit", type=int, default=32, help="Number of tokens/ids in trace previews.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional tokenizer trace log file.")
    parser.add_argument("--format", choices=("pretty", "json"), default="pretty", help="Output format.")
    parser.add_argument("--output", type=Path, default=None, help="Optional report output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(trace=args.trace, log_file=args.log_file)
    report = inspect_tokenizer(args)
    LOGGER.info(
        "inspect_tokenizer.output | tokens=%d | ids=%s | format=%s",
        report["output"]["token_count"],
        report["output"]["id_count"],
        args.format,
    )
    write_report(report, output_format=args.format, output_path=args.output)


if __name__ == "__main__":
    main()
