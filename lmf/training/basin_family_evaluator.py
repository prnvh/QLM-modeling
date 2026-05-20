"""Evaluate basin-family separation after field-loop readout (Commit D3)."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.basin.basin_family_contrastive import BasinFamilyContrastiveMetrics  # noqa: E402
from lmf.training.basin_families import BasinFamilyExample, load_basin_families  # noqa: E402
from lmf.training.basin_field_stack import BasinFieldStack, build_basin_field_stack  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_BASIN_FAMILIES_PATH = PROJECT_ROOT / "data" / "stage1" / "basin_families.jsonl"


@dataclass(frozen=True)
class BasinFamilyEvalReport:
    contrastive_loss: float
    same_family_cosine_mean: float
    different_family_cosine_mean: float
    separation_margin: float
    num_examples: int
    num_pairs_same: int
    num_pairs_different: int

    @property
    def separation_ok(self) -> bool:
        return self.separation_margin > 0.0


def evaluate_basin_families(
    examples: list[BasinFamilyExample],
    *,
    stack: BasinFieldStack | None = None,
    cue_dim: int = 16,
    num_traces: int = 64,
    top_k: int = 16,
    num_basins: int = 16,
    seed: int = 7,
) -> BasinFamilyEvalReport:
    if len(examples) < 2:
        raise ValueError("need at least two examples to evaluate family separation.")

    model = stack or build_basin_field_stack(
        examples,
        cue_dim=cue_dim,
        num_traces=num_traces,
        top_k=top_k,
        num_basins=num_basins,
        seed=seed,
    )
    model.eval()
    with torch.no_grad():
        loss, metrics = model.contrastive_loss_from_batch(examples)

    return BasinFamilyEvalReport(
        contrastive_loss=float(loss.detach().item()),
        same_family_cosine_mean=metrics.same_family_cosine_mean,
        different_family_cosine_mean=metrics.different_family_cosine_mean,
        separation_margin=metrics.separation_margin,
        num_examples=len(examples),
        num_pairs_same=metrics.num_pairs_same,
        num_pairs_different=metrics.num_pairs_different,
    )


def format_basin_family_report(*, title: str, report: BasinFamilyEvalReport) -> str:
    lines = [
        title,
        f"examples: {report.num_examples}",
        f"contrastive_loss: {report.contrastive_loss:.6f}",
        f"same_family_cosine_mean: {report.same_family_cosine_mean:.4f}",
        f"different_family_cosine_mean: {report.different_family_cosine_mean:.4f}",
        f"separation_margin: {report.separation_margin:.4f}",
        f"pairs_same: {report.num_pairs_same} pairs_different: {report.num_pairs_different}",
        f"separation_ok: {report.separation_ok}",
    ]
    return "\n".join(lines)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate basin-family contrastive separation.")
    parser.add_argument(
        "text",
        nargs="*",
        help='Optional "text" prefix plus sentence; if omitted, evaluates the full dataset file.',
    )
    parser.add_argument("--families-file", type=str, default=str(DEFAULT_BASIN_FAMILIES_PATH))
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--num-basins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if args.trace:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    if args.text:
        parts = args.text[1:] if args.text[0].lower() == "text" else args.text
        sentence = " ".join(parts)
        examples = load_basin_families(args.families_file)
        match = [example for example in examples if example.text == sentence]
        if not match:
            raise ValueError(f"sentence not found in families file: {sentence!r}")
        report = evaluate_basin_families(
            match if len(match) >= 2 else examples,
            cue_dim=args.cue_dim,
            num_traces=args.num_traces,
            top_k=args.top_k,
            num_basins=args.num_basins,
            seed=args.seed,
        )
        title = f"Basin families (query in file; eval batch n={report.num_examples})"
    else:
        examples = load_basin_families(args.families_file)
        report = evaluate_basin_families(
            examples,
            cue_dim=args.cue_dim,
            num_traces=args.num_traces,
            top_k=args.top_k,
            num_basins=args.num_basins,
            seed=args.seed,
        )
        title = "Basin families (full dataset)"

    print(format_basin_family_report(title=title, report=report))


if __name__ == "__main__":
    main()
