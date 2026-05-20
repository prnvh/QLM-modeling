"""Train basin-family contrastive objective through the full field loop (Commit D3)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.training.basin_families import load_basin_families  # noqa: E402
from lmf.training.basin_family_evaluator import (  # noqa: E402
    evaluate_basin_families,
    format_basin_family_report,
)
from lmf.training.basin_field_stack import build_basin_field_stack  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_BASIN_FAMILIES_PATH = PROJECT_ROOT / "data" / "stage1" / "basin_families.jsonl"


def train_basin_families(
    *,
    families_file: str | Path,
    steps: int = 300,
    lr: float = 1e-3,
    num_traces: int = 64,
    top_k: int = 16,
    cue_dim: int = 16,
    num_basins: int = 16,
    settling_steps: int = 3,
    contrastive_weight: float = 0.05,
    seed: int = 7,
    trace: bool = False,
) -> None:
    examples = load_basin_families(families_file)
    stack = build_basin_field_stack(
        examples,
        cue_dim=cue_dim,
        num_traces=num_traces,
        top_k=top_k,
        num_basins=num_basins,
        settling_steps=settling_steps,
        contrastive_weight=contrastive_weight,
        seed=seed,
    )
    optimizer = torch.optim.Adam(stack.parameters(), lr=lr)

    for step in range(1, steps + 1):
        stack.train()
        optimizer.zero_grad()
        loss, metrics = stack.contrastive_loss_from_batch(examples)
        if float(loss.detach()) == 0.0 and not metrics.num_pairs_same:
            raise RuntimeError("contrastive loss is zero with no same-family pairs in batch.")
        loss.backward()
        optimizer.step()

        if trace or step == 1 or step == steps or step % max(steps // 5, 1) == 0:
            LOGGER.info(
                "train_basin_families step=%s loss=%.6f same_cos=%.4f diff_cos=%.4f margin=%.4f",
                step,
                float(loss.detach().item()),
                metrics.same_family_cosine_mean,
                metrics.different_family_cosine_mean,
                metrics.separation_margin,
            )

    report = evaluate_basin_families(examples, stack=stack)
    print(format_basin_family_report(title="TRAINED SUMMARY", report=report))


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train basin-family contrastive loss on field readout.")
    parser.add_argument("--families-file", type=str, default=str(DEFAULT_BASIN_FAMILIES_PATH))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--num-basins", type=int, default=16)
    parser.add_argument("--settling-steps", type=int, default=3)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if args.trace:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    train_basin_families(
        families_file=args.families_file,
        steps=args.steps,
        lr=args.lr,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        num_basins=args.num_basins,
        settling_steps=args.settling_steps,
        contrastive_weight=args.contrastive_weight,
        seed=args.seed,
        trace=args.trace,
    )


if __name__ == "__main__":
    main()
