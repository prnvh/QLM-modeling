"""Load and validate cue-pair binding supervision edges (Commit C3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lmf.core.input.tokenizer import SimpleTokenizer, build_vocabulary_from_texts, regex_tokenize


@dataclass(frozen=True)
class BindingEdge:
    """One labeled cue pair inside a prompt."""

    cue_a: str
    cue_b: str
    label: int

    def __post_init__(self) -> None:
        if not self.cue_a or not self.cue_b:
            raise ValueError("cue_a and cue_b must be non-empty.")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 (negative) or 1 (positive).")


@dataclass(frozen=True)
class BindingEdgeExample:
    """Prompt plus labeled cue-pair edges for binding supervision."""

    text: str
    edges: tuple[BindingEdge, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must be non-empty.")
        if not self.edges:
            raise ValueError("edges must not be empty.")


def _normalize_token(token: str) -> str:
    return token.strip().lower()


def _parse_edge(raw: object) -> BindingEdge:
    if not isinstance(raw, dict):
        raise ValueError("each edge must be a JSON object.")
    cue_a = raw.get("cue_a")
    cue_b = raw.get("cue_b")
    label = raw.get("label")
    if not isinstance(cue_a, str) or not isinstance(cue_b, str):
        raise ValueError("edge cue_a and cue_b must be strings.")
    if not isinstance(label, int):
        raise ValueError("edge label must be an integer (0 or 1).")
    return BindingEdge(cue_a=_normalize_token(cue_a), cue_b=_normalize_token(cue_b), label=label)


def _parse_example(raw: object, *, line_number: int) -> BindingEdgeExample:
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_number}: example must be a JSON object.")

    text = raw.get("text")
    edges_raw = raw.get("edges")
    if not isinstance(text, str):
        raise ValueError(f"line {line_number}: text must be a string.")
    if not isinstance(edges_raw, list) or not edges_raw:
        raise ValueError(f"line {line_number}: edges must be a non-empty list.")

    edges = tuple(_parse_edge(edge) for edge in edges_raw)
    return BindingEdgeExample(text=text.strip(), edges=edges)


def tokens_in_text(text: str, *, lowercase: bool = True) -> set[str]:
    return {_normalize_token(token) for token in regex_tokenize(text, lowercase=lowercase)}


def validate_example_present_cues(example: BindingEdgeExample) -> None:
    """Ensure both cues in every edge appear in the prompt tokens."""

    present = tokens_in_text(example.text)
    missing: list[str] = []
    for edge in example.edges:
        if edge.cue_a not in present:
            missing.append(edge.cue_a)
        if edge.cue_b not in present:
            missing.append(edge.cue_b)
    if missing:
        unique = sorted(set(missing))
        raise ValueError(
            f"edge cues must appear in the prompt; missing from {example.text!r}: {unique}"
        )


def load_binding_edges(
    path: str | Path,
    *,
    require_present_cues: bool = True,
) -> list[BindingEdgeExample]:
    """Load binding edge examples from a JSONL file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"binding edges file not found: {file_path}")

    examples: list[BindingEdgeExample] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            example = _parse_example(json.loads(stripped), line_number=line_number)
            if require_present_cues:
                validate_example_present_cues(example)
            examples.append(example)

    if not examples:
        raise ValueError(f"binding edges file is empty: {file_path}")
    return examples


@dataclass(frozen=True)
class ResolvedBindingEdge:
    """Cue-pair label anchored to token positions in the encoded sequence."""

    cue_a: str
    cue_b: str
    label: int
    cue_a_positions: tuple[int, ...]
    cue_b_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.cue_a_positions or not self.cue_b_positions:
            raise ValueError("resolved edge must include at least one position per cue.")


def encoded_token_positions(text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[tuple[int, str]]:
    """Return ``(sequence_index, token)`` for the encoded prompt."""

    tokenizer = build_example_tokenizer(text)
    ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    return [(index, _normalize_token(tokenizer.vocab.token(token_id))) for index, token_id in enumerate(ids)]


def positions_for_cue(token_rows: list[tuple[int, str]], cue: str) -> tuple[int, ...]:
    target = _normalize_token(cue)
    return tuple(index for index, token in token_rows if token == target)


def resolve_example_edges(example: BindingEdgeExample) -> tuple[ResolvedBindingEdge, ...]:
    """Map string-labeled edges to sequence positions (supports repeated words)."""

    token_rows = encoded_token_positions(example.text)
    resolved: list[ResolvedBindingEdge] = []
    for edge in example.edges:
        cue_a_positions = positions_for_cue(token_rows, edge.cue_a)
        cue_b_positions = positions_for_cue(token_rows, edge.cue_b)
        if not cue_a_positions or not cue_b_positions:
            raise ValueError(
                f"could not resolve edge ({edge.cue_a!r}, {edge.cue_b!r}) in text {example.text!r}"
            )
        resolved.append(
            ResolvedBindingEdge(
                cue_a=edge.cue_a,
                cue_b=edge.cue_b,
                label=edge.label,
                cue_a_positions=cue_a_positions,
                cue_b_positions=cue_b_positions,
            )
        )
    return tuple(resolved)


def build_example_tokenizer(text: str) -> SimpleTokenizer:
    """Build the same style of temporary vocab used by route_text inspection."""

    vocab = build_vocabulary_from_texts([text])
    return SimpleTokenizer(vocab=vocab)


__all__ = [
    "BindingEdge",
    "BindingEdgeExample",
    "ResolvedBindingEdge",
    "build_example_tokenizer",
    "encoded_token_positions",
    "load_binding_edges",
    "positions_for_cue",
    "resolve_example_edges",
    "tokens_in_text",
    "validate_example_present_cues",
]
