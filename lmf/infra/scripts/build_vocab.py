"""Build a Stage 1 vocabulary from local text files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.input.tokenizer import (  # noqa: E402
    DEFAULT_TOKEN_PATTERN,
    SPECIAL_TOKENS,
    Vocabulary,
    regex_tokenize,
)

LOGGER = logging.getLogger("lmf.build_vocab")


def configure_logging(*, verbose: bool, log_file: Path | None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(levelname)s %(message)s")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=handlers, force=True)


def preview_tokens(tokens: list[str], *, limit: int) -> str:
    if limit < 0:
        raise ValueError("preview token limit must be non-negative.")
    shown = [repr(token) for token in tokens[:limit]]
    if len(tokens) > limit:
        suffix = f"{', ' if shown else ''}... (+{len(tokens) - limit})"
    else:
        suffix = ""
    return "[" + ", ".join(shown) + suffix + "]"


def iter_text_files(paths: Iterable[Path], *, glob: str) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(candidate for candidate in path.rglob(glob) if candidate.is_file()))
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(dict.fromkeys(files))


def count_tokens(
    files: Iterable[Path],
    *,
    lowercase: bool,
    encoding: str,
    pattern: str,
    preview_count: int,
) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    total_tokens = 0
    for file_path in files:
        text = file_path.read_text(encoding=encoding)
        tokens = regex_tokenize(text, lowercase=lowercase, pattern=pattern)
        counts.update(tokens)
        total_tokens += len(tokens)
        LOGGER.info(
            "build_vocab.tokenized_file | path=%s | chars=%d | tokens=%d | preview=%s",
            file_path,
            len(text),
            len(tokens),
            preview_tokens(tokens, limit=preview_count),
        )
    return counts, total_tokens


def build_vocab_from_counts(
    counts: Counter[str],
    *,
    min_freq: int,
    max_size: int | None,
) -> Vocabulary:
    if min_freq < 1:
        raise ValueError("--min-freq must be at least 1.")
    if max_size is not None and max_size < len(SPECIAL_TOKENS):
        raise ValueError("--max-size must leave room for all special tokens.")

    token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    candidates = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_freq and token not in token_to_id
    ]
    if max_size is not None:
        candidates = candidates[: max_size - len(token_to_id)]

    for token in candidates:
        token_to_id[token] = len(token_to_id)
    return Vocabulary(token_to_id=token_to_id)


def write_vocab(
    vocab: Vocabulary,
    output_path: Path,
    *,
    counts: Counter[str],
    num_files: int,
    total_tokens: int,
    min_freq: int,
    max_size: int | None,
    lowercase: bool,
    pattern: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = vocab.to_dict()
    payload["metadata"] = {
        "num_files": num_files,
        "total_tokens": total_tokens,
        "vocab_size": len(vocab),
        "min_freq": min_freq,
        "max_size": max_size,
        "lowercase": lowercase,
        "pattern": pattern,
    }
    special_tokens = set(vocab.special_tokens)
    payload["token_counts"] = {
        token: counts[token]
        for token in vocab.id_to_token
        if token not in special_tokens
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Text files or directories to read.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/vocab.json"),
        help="Path for the output vocabulary JSON.",
    )
    parser.add_argument("--glob", default="*.txt", help="Glob used when an input is a directory.")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum token frequency to keep.")
    parser.add_argument("--max-size", type=int, default=None, help="Maximum vocabulary size.")
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Keep original casing instead of lowercasing text.",
    )
    parser.add_argument("--encoding", default="utf-8", help="Input text file encoding.")
    parser.add_argument("--pattern", default=DEFAULT_TOKEN_PATTERN, help="Regex token pattern.")
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=20,
        help="Number of token/count entries to include in human-readable logs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable more detailed logs.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional human-readable log file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose, log_file=args.log_file)
    if args.preview_tokens < 0:
        raise ValueError("--preview-tokens must be non-negative.")
    LOGGER.info(
        "build_vocab.start | inputs=%s | output=%s | glob=%s | lowercase=%s",
        [str(path) for path in args.inputs],
        args.output,
        args.glob,
        not args.case_sensitive,
    )

    files = iter_text_files(args.inputs, glob=args.glob)
    if not files:
        raise ValueError("No input text files found.")

    LOGGER.info(
        "build_vocab.files | count=%d | files=%s",
        len(files),
        [str(path) for path in files],
    )
    lowercase = not args.case_sensitive
    counts, total_tokens = count_tokens(
        files,
        lowercase=lowercase,
        encoding=args.encoding,
        pattern=args.pattern,
        preview_count=args.preview_tokens,
    )
    LOGGER.info(
        "build_vocab.counts | total_tokens=%d | unique_tokens=%d | top=%s",
        total_tokens,
        len(counts),
        counts.most_common(args.preview_tokens),
    )
    vocab = build_vocab_from_counts(counts, min_freq=args.min_freq, max_size=args.max_size)
    LOGGER.info(
        "build_vocab.vocabulary | vocab_size=%d | special_tokens=%s | first_tokens=%s",
        len(vocab),
        list(vocab.special_tokens),
        vocab.id_to_token[: args.preview_tokens],
    )
    write_vocab(
        vocab,
        args.output,
        counts=counts,
        num_files=len(files),
        total_tokens=total_tokens,
        min_freq=args.min_freq,
        max_size=args.max_size,
        lowercase=lowercase,
        pattern=args.pattern,
    )
    LOGGER.info("build_vocab.output | path=%s", args.output)
    print(f"Wrote {len(vocab)} tokens from {len(files)} file(s) to {args.output}")


if __name__ == "__main__":
    main()
