"""Inspectable tokenizer and vocabulary utilities 

This module intentionally stays simpler than BPE/WordPiece tokenizers. It
provides regex tokenization, stable vocabulary building, and small encode/decode
helpers so early training data remains easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS: tuple[str, ...] = (
    PAD_TOKEN,
    UNK_TOKEN,
    MASK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)

DEFAULT_TOKEN_PATTERN = r"\w+(?:['-]\w+)*|[^\w\s]"
DEFAULT_TRACE_LIMIT = 32

LOGGER = logging.getLogger(__name__)


def _preview_text(text: str, *, limit: int) -> str:
    if limit < 0:
        raise ValueError("preview text limit must be non-negative.")
    compact = text.replace("\n", "\\n").replace("\t", "\\t")
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _preview_sequence(values: Sequence[object], *, limit: int) -> str:
    if limit < 0:
        raise ValueError("preview sequence limit must be non-negative.")
    shown = [repr(value) for value in values[:limit]]
    if len(values) > limit:
        suffix = f"{', ' if shown else ''}... (+{len(values) - limit})"
    else:
        suffix = ""
    return "[" + ", ".join(shown) + suffix + "]"


def _log_trace(
    logger: logging.Logger,
    enabled: bool,
    event: str,
    **fields: object,
) -> None:
    if not enabled:
        return

    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s%s", event, f" | {details}" if details else "")


def regex_tokenize(
    text: str,
    *,
    lowercase: bool = True,
    pattern: str = DEFAULT_TOKEN_PATTERN,
) -> list[str]:
    """Split text into word-like tokens and punctuation tokens."""

    if lowercase:
        text = text.lower()
    return re.findall(pattern, text, flags=re.UNICODE)


def detokenize(tokens: Sequence[str]) -> str:
    """Join tokens into readable text with simple punctuation spacing."""

    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    return text.strip()


@dataclass
class Vocabulary:
    """Token-to-id mapping with stable special token ids."""

    token_to_id: Mapping[str, int]
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS
    unk_token: str = UNK_TOKEN
    id_to_token: list[str] = field(init=False)

    def __post_init__(self) -> None:
        token_to_id = dict(self.token_to_id)
        if self.unk_token not in token_to_id:
            raise ValueError(f"Vocabulary must include unk token {self.unk_token!r}.")

        size = len(token_to_id)
        id_to_token: list[str | None] = [None] * size
        for token, token_id in token_to_id.items():
            if token_id < 0 or token_id >= size:
                raise ValueError(f"Token id for {token!r} is out of range: {token_id}.")
            if id_to_token[token_id] is not None:
                raise ValueError(f"Duplicate token id detected: {token_id}.")
            id_to_token[token_id] = token

        missing = [idx for idx, token in enumerate(id_to_token) if token is None]
        if missing:
            raise ValueError(f"Vocabulary ids must be contiguous; missing ids: {missing}.")

        self.token_to_id = token_to_id
        self.id_to_token = [token for token in id_to_token if token is not None]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[MASK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def token_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def token(self, token_id: int) -> str:
        if token_id < 0 or token_id >= len(self.id_to_token):
            raise IndexError(f"Token id out of range: {token_id}.")
        return self.id_to_token[token_id]

    def encode(
        self,
        tokens: Iterable[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
        pad_to_length: int | None = None,
    ) -> list[int]:
        ids = [self.token_id(token) for token in tokens]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None:
            ids = ids[:max_length]
        if pad_to_length is not None:
            if len(ids) > pad_to_length:
                ids = ids[:pad_to_length]
            ids = ids + [self.pad_id] * (pad_to_length - len(ids))
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = False) -> list[str]:
        tokens = [self.token(int(token_id)) for token_id in token_ids]
        if skip_special:
            special = set(self.special_tokens)
            tokens = [token for token in tokens if token not in special]
        return tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "token_to_id": dict(self.token_to_id),
            "special_tokens": list(self.special_tokens),
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Vocabulary":
        token_to_id = data.get("token_to_id")
        if not isinstance(token_to_id, Mapping):
            raise ValueError("Vocabulary data must include a token_to_id mapping.")

        raw_special_tokens = data.get("special_tokens", SPECIAL_TOKENS)
        if isinstance(raw_special_tokens, str) or not isinstance(raw_special_tokens, Sequence):
            raise ValueError("special_tokens must be a sequence of token strings.")

        special_tokens = tuple(str(token) for token in raw_special_tokens)
        unk_token = str(data.get("unk_token", UNK_TOKEN))
        return cls(
            token_to_id={str(token): int(token_id) for token, token_id in token_to_id.items()},
            special_tokens=special_tokens,
            unk_token=unk_token,
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class SimpleTokenizer:
    """Regex tokenizer with optional vocabulary-backed encoding."""

    vocab: Vocabulary | None = None
    lowercase: bool = True
    pattern: str = DEFAULT_TOKEN_PATTERN
    trace: bool = False
    trace_limit: int = DEFAULT_TRACE_LIMIT
    logger: logging.Logger = field(default=LOGGER, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.trace_limit < 0:
            raise ValueError("trace_limit must be non-negative.")

    def tokenize(self, text: str) -> list[str]:
        tokens = regex_tokenize(text, lowercase=self.lowercase, pattern=self.pattern)
        _log_trace(
            self.logger,
            self.trace,
            "tokenizer.tokenize",
            lowercase=self.lowercase,
            input=_preview_text(text, limit=self.trace_limit * 4),
            token_count=len(tokens),
            tokens=_preview_sequence(tokens, limit=self.trace_limit),
        )
        return tokens

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
        pad_to_length: int | None = None,
    ) -> list[int]:
        if self.vocab is None:
            raise ValueError("SimpleTokenizer.encode requires a vocabulary.")
        tokens = self.tokenize(text)
        _log_trace(
            self.logger,
            self.trace,
            "tokenizer.encode.tokens",
            add_bos=add_bos,
            add_eos=add_eos,
            max_length=max_length,
            pad_to_length=pad_to_length,
            tokens=_preview_sequence(tokens, limit=self.trace_limit),
        )
        ids = self.vocab.encode(
            tokens,
            add_bos=add_bos,
            add_eos=add_eos,
            max_length=max_length,
            pad_to_length=pad_to_length,
        )
        _log_trace(
            self.logger,
            self.trace,
            "tokenizer.encode.ids",
            id_count=len(ids),
            ids=_preview_sequence(ids, limit=self.trace_limit),
        )
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = False) -> str:
        if self.vocab is None:
            raise ValueError("SimpleTokenizer.decode requires a vocabulary.")
        ids = [int(token_id) for token_id in token_ids]
        tokens = self.vocab.decode(ids, skip_special=skip_special)
        text = detokenize(tokens)
        _log_trace(
            self.logger,
            self.trace,
            "tokenizer.decode",
            skip_special=skip_special,
            ids=_preview_sequence(ids, limit=self.trace_limit),
            tokens=_preview_sequence(tokens, limit=self.trace_limit),
            text=_preview_text(text, limit=self.trace_limit * 4),
        )
        return text


def build_vocabulary(
    token_sequences: Iterable[Iterable[str]],
    *,
    min_freq: int = 1,
    max_size: int | None = None,
    special_tokens: Sequence[str] = SPECIAL_TOKENS,
) -> Vocabulary:
    """Build a stable vocabulary from token sequences.

    Tokens are sorted by descending frequency and then alphabetically so repeated
    runs over the same text produce identical ids.
    """

    if min_freq < 1:
        raise ValueError("min_freq must be at least 1.")
    if max_size is not None and max_size < len(special_tokens):
        raise ValueError("max_size must leave room for all special tokens.")

    counts: Counter[str] = Counter()
    for tokens in token_sequences:
        counts.update(tokens)

    token_to_id: dict[str, int] = {}
    for token in special_tokens:
        if token in token_to_id:
            raise ValueError(f"Duplicate special token: {token!r}.")
        token_to_id[token] = len(token_to_id)

    candidates = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_freq and token not in token_to_id
    ]
    if max_size is not None:
        candidates = candidates[: max_size - len(token_to_id)]

    for token in candidates:
        token_to_id[token] = len(token_to_id)

    return Vocabulary(token_to_id=token_to_id, special_tokens=tuple(special_tokens))


def build_vocabulary_from_texts(
    texts: Iterable[str],
    *,
    lowercase: bool = True,
    min_freq: int = 1,
    max_size: int | None = None,
    pattern: str = DEFAULT_TOKEN_PATTERN,
) -> Vocabulary:
    token_sequences = (
        regex_tokenize(text, lowercase=lowercase, pattern=pattern)
        for text in texts
    )
    return build_vocabulary(token_sequences, min_freq=min_freq, max_size=max_size)


def inspect_text(text: str, *, lowercase: bool = True) -> dict[str, object]:
    """Return a compact, human-readable tokenizer inspection packet."""

    vocab = build_vocabulary_from_texts([text], lowercase=lowercase)
    tokenizer = SimpleTokenizer(vocab=vocab, lowercase=lowercase)
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    return {
        "input": text,
        "tokens": tokens,
        "ids": ids,
        "token_ids": [(idx, token_id, vocab.token(token_id)) for idx, token_id in enumerate(ids)],
        "decoded": tokenizer.decode(ids, skip_special=True),
    }


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show tokenizer input, tokens, ids, and decoded text.")
    parser.add_argument(
        "text",
        nargs="+",
        help='Text to inspect. You may optionally start with the word "text".',
    )
    parser.add_argument("--case-sensitive", action="store_true", help="Keep original casing.")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)
    report = inspect_text(text, lowercase=not args.case_sensitive)

    print(f"Input: {report['input']}")
    print(f"Tokens: {report['tokens']}")
    print(f"IDs: {report['ids']}")
    print("Token ID rows:")
    for position, token_id, token in report["token_ids"]:
        print(f"  {position}: {token_id} -> {token}")
    print(f"Decoded: {report['decoded']}")


__all__ = [
    "BOS_TOKEN",
    "DEFAULT_TOKEN_PATTERN",
    "EOS_TOKEN",
    "MASK_TOKEN",
    "PAD_TOKEN",
    "DEFAULT_TRACE_LIMIT",
    "SPECIAL_TOKENS",
    "UNK_TOKEN",
    "SimpleTokenizer",
    "Vocabulary",
    "build_vocabulary",
    "build_vocabulary_from_texts",
    "detokenize",
    "inspect_text",
    "regex_tokenize",
]


if __name__ == "__main__":
    main()
