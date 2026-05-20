"""Load and validate basin-family supervision examples (Commit D3)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_FAMILY_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class BasinFamilyExample:
    """One prompt labeled with a relational basin family."""

    text: str
    family: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must be non-empty.")
        if not self.family.strip():
            raise ValueError("family must be non-empty.")
        if not _FAMILY_PATTERN.match(self.family):
            raise ValueError(
                f"family must be lowercase slug form (e.g. financial-bank); got {self.family!r}."
            )


def _parse_example(raw: object, *, line_number: int) -> BasinFamilyExample:
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_number}: example must be a JSON object.")

    text = raw.get("text")
    family = raw.get("family")
    if not isinstance(text, str):
        raise ValueError(f"line {line_number}: text must be a string.")
    if not isinstance(family, str):
        raise ValueError(f"line {line_number}: family must be a string.")

    return BasinFamilyExample(text=text.strip(), family=family.strip().lower())


def validate_family_coverage(examples: list[BasinFamilyExample], *, min_per_family: int = 2) -> None:
    """Ensure each family has enough examples for contrastive learning."""

    counts: dict[str, int] = {}
    for example in examples:
        counts[example.family] = counts.get(example.family, 0) + 1

    too_small = [family for family, count in counts.items() if count < min_per_family]
    if too_small:
        raise ValueError(
            "each basin family needs at least "
            f"{min_per_family} examples for contrastive training; short: {sorted(too_small)}"
        )


def load_basin_families(
    path: str | Path,
    *,
    min_per_family: int = 2,
) -> list[BasinFamilyExample]:
    """Load basin family examples from JSONL."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"basin families file not found: {file_path}")

    examples: list[BasinFamilyExample] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            examples.append(_parse_example(json.loads(stripped), line_number=line_number))

    if not examples:
        raise ValueError(f"basin families file is empty: {file_path}")

    validate_family_coverage(examples, min_per_family=min_per_family)
    return examples


def build_family_registry(examples: list[BasinFamilyExample]) -> dict[str, int]:
    """Stable family name → integer id (sorted for repeatability)."""

    families = sorted({example.family for example in examples})
    return {family: index for index, family in enumerate(families)}


__all__ = [
    "BasinFamilyExample",
    "build_family_registry",
    "load_basin_families",
    "validate_family_coverage",
]
