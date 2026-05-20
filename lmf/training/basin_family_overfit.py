"""Basin-family contrastive overfit sanity (Commit D3 guard + causal ablations)."""

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

from lmf.training.basin_families import BasinFamilyExample  # noqa: E402
from lmf.training.basin_family_evaluator import (  # noqa: E402
    BasinFamilyEvalReport,
    evaluate_basin_families,
)
from lmf.training.basin_field_stack import BasinFieldStack, build_basin_field_stack  # noqa: E402

LOGGER = logging.getLogger(__name__)

OVERFIT_MIN_SEPARATION = 0.15
OVERFIT_MAX_DIFF_COSINE = 0.85
OVERFIT_MIN_CAUSAL_MARGIN_DROP = 0.03


@dataclass(frozen=True)
class BasinFamilyOverfitConfig:
    steps: int = 500
    lr: float = 5e-3
    num_traces: int = 32
    top_k: int = 12
    cue_dim: int = 16
    num_basins: int = 12
    settling_steps: int = 2
    contrastive_weight: float = 0.05
    seed: int = 7


@dataclass(frozen=True)
class BasinFamilyOverfitGateResult:
    passed: bool
    failures: tuple[str, ...]
    trained_report: BasinFamilyEvalReport
    untrained_report: BasinFamilyEvalReport
    no_bound_pair_margin: float
    permuted_vectors_margin: float

    def to_log_fields(self) -> dict[str, str]:
        return {
            "passed": str(self.passed),
            "trained_margin": f"{self.trained_report.separation_margin:.4f}",
            "untrained_margin": f"{self.untrained_report.separation_margin:.4f}",
            "no_bound_pair_margin": f"{self.no_bound_pair_margin:.4f}",
            "permuted_vectors_margin": f"{self.permuted_vectors_margin:.4f}",
            "failures": ",".join(self.failures) if self.failures else "none",
        }


def default_overfit_examples() -> list[BasinFamilyExample]:
    return [
        BasinFamilyExample("alpha beta one", "family-a"),
        BasinFamilyExample("alpha beta two", "family-a"),
        BasinFamilyExample("gamma delta one", "family-b"),
        BasinFamilyExample("gamma delta two", "family-b"),
        BasinFamilyExample("epsilon zeta one", "family-c"),
        BasinFamilyExample("epsilon zeta two", "family-c"),
    ]


def train_stack(
    stack: BasinFieldStack,
    examples: list[BasinFamilyExample],
    *,
    steps: int,
    lr: float,
    trace: bool,
) -> None:
    optimizer = torch.optim.Adam(stack.parameters(), lr=lr)
    for step in range(1, steps + 1):
        stack.train()
        optimizer.zero_grad()
        loss, metrics = stack.contrastive_loss_from_batch(examples)
        loss.backward()
        optimizer.step()
        if trace and (step == 1 or step == steps or step % max(steps // 5, 1) == 0):
            LOGGER.info(
                "basin_family_overfit step=%s loss=%.6f margin=%.4f",
                step,
                float(loss.detach().item()),
                metrics.separation_margin,
            )


def evaluate_overfit_gates(
    report: BasinFamilyEvalReport,
    *,
    trained_margin: float,
    no_bound_pair_margin: float,
    permuted_vectors_margin: float,
    untrained_margin: float,
) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    if report.separation_margin < OVERFIT_MIN_SEPARATION:
        failures.append(
            f"separation_margin {report.separation_margin:.4f} < {OVERFIT_MIN_SEPARATION}"
        )
    if report.different_family_cosine_mean > OVERFIT_MAX_DIFF_COSINE:
        failures.append(
            f"different_family_cosine_mean {report.different_family_cosine_mean:.4f} "
            f"> {OVERFIT_MAX_DIFF_COSINE}"
        )
    if not report.separation_ok:
        failures.append("separation_ok is false")
    if untrained_margin >= trained_margin:
        failures.append(
            f"untrained margin {untrained_margin:.4f} must be below trained {trained_margin:.4f}"
        )
    if trained_margin - no_bound_pair_margin < OVERFIT_MIN_CAUSAL_MARGIN_DROP:
        failures.append(
            f"zero bound-pair path must hurt: full {trained_margin:.4f} vs "
            f"no_bound_pair {no_bound_pair_margin:.4f} (need drop >= {OVERFIT_MIN_CAUSAL_MARGIN_DROP})"
        )
    if trained_margin - permuted_vectors_margin < OVERFIT_MIN_CAUSAL_MARGIN_DROP:
        failures.append(
            f"permuted basin_bank vectors must hurt: full {trained_margin:.4f} vs "
            f"permuted {permuted_vectors_margin:.4f} (need drop >= {OVERFIT_MIN_CAUSAL_MARGIN_DROP})"
        )
    return (not failures, tuple(failures))


def run_causal_ablation_margins(
    stack: BasinFieldStack,
    examples: list[BasinFamilyExample],
) -> tuple[float, float]:
    """Return (margin with bound_pair=0, margin with permuted basin_bank vectors)."""

    bound_scale = stack.field_loop.binding_forces.config.bound_pair_to_basin_scale
    stack.set_bound_pair_to_basin_scale(0.0)
    no_bound_margin = stack.separation_margin_for_examples(examples)
    stack.set_bound_pair_to_basin_scale(bound_scale)

    vectors = stack.basin_vectors().detach().clone()
    perm = torch.randperm(vectors.shape[0], device=vectors.device)
    permuted_vectors_margin = stack.separation_margin_for_examples(
        examples,
        basin_vectors=vectors[perm],
    )

    return no_bound_margin, permuted_vectors_margin


def run_basin_family_overfit_sanity(
    *,
    config: BasinFamilyOverfitConfig | None = None,
    trace: bool = False,
) -> BasinFamilyOverfitGateResult:
    config = config or BasinFamilyOverfitConfig()
    examples = default_overfit_examples()

    untrained = build_basin_field_stack(
        examples,
        cue_dim=config.cue_dim,
        num_traces=config.num_traces,
        top_k=config.top_k,
        num_basins=config.num_basins,
        settling_steps=config.settling_steps,
        contrastive_weight=config.contrastive_weight,
        seed=config.seed,
    )
    untrained_report = evaluate_basin_families(examples, stack=untrained)

    stack = build_basin_field_stack(
        examples,
        cue_dim=config.cue_dim,
        num_traces=config.num_traces,
        top_k=config.top_k,
        num_basins=config.num_basins,
        settling_steps=config.settling_steps,
        contrastive_weight=config.contrastive_weight,
        seed=config.seed,
    )
    train_stack(stack, examples, steps=config.steps, lr=config.lr, trace=trace)
    trained_report = evaluate_basin_families(examples, stack=stack)

    no_bound_margin, permuted_vectors_margin = run_causal_ablation_margins(stack, examples)

    passed, failures = evaluate_overfit_gates(
        trained_report,
        trained_margin=trained_report.separation_margin,
        no_bound_pair_margin=no_bound_margin,
        permuted_vectors_margin=permuted_vectors_margin,
        untrained_margin=untrained_report.separation_margin,
    )

    return BasinFamilyOverfitGateResult(
        passed=passed,
        failures=failures,
        trained_report=trained_report,
        untrained_report=untrained_report,
        no_bound_pair_margin=no_bound_margin,
        permuted_vectors_margin=permuted_vectors_margin,
    )


def format_basin_family_overfit_report(result: BasinFamilyOverfitGateResult) -> str:
    lines = [
        f"passed: {result.passed}",
        f"failures: {', '.join(result.failures) if result.failures else 'none'}",
        "",
        "untrained:",
        f"  margin={result.untrained_report.separation_margin:.4f}",
        f"  same_cos={result.untrained_report.same_family_cosine_mean:.4f}",
        f"  diff_cos={result.untrained_report.different_family_cosine_mean:.4f}",
        "",
        "trained (full field path):",
        f"  margin={result.trained_report.separation_margin:.4f}",
        f"  same_cos={result.trained_report.same_family_cosine_mean:.4f}",
        f"  diff_cos={result.trained_report.different_family_cosine_mean:.4f}",
        "",
        "causal ablations (same weights, eval only):",
        f"  no_bound_pair margin={result.no_bound_pair_margin:.4f}",
        f"  permuted_vectors margin={result.permuted_vectors_margin:.4f}",
    ]
    return "\n".join(lines)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basin-family contrastive overfit sanity.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if args.trace:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    result = run_basin_family_overfit_sanity(
        config=BasinFamilyOverfitConfig(steps=args.steps),
        trace=args.trace,
    )
    print(format_basin_family_overfit_report(result))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
