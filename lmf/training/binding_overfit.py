"""Binding overfit sanity checks (Commit C5).

Trains a tiny synthetic edge set end-to-end and verifies the binding stack can
memorize labels while a no-binding ablation stays worse. Gates match the commit
plan and are runnable from CLI for manual inspection.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.training.binding_edge_evaluator import (  # noqa: E402
    BindingEdgeEvalReport,
    evaluate_binding_edges,
)
from lmf.training.binding_edges import BindingEdge, BindingEdgeExample  # noqa: E402
from lmf.training.binding_stack import BindingStack, build_binding_stack  # noqa: E402

LOGGER = logging.getLogger(__name__)

OVERFIT_MIN_ACCURACY = 0.90
OVERFIT_MIN_POSITIVE_MASS = 0.70
OVERFIT_MAX_NEGATIVE_MASS = 0.20


@dataclass(frozen=True)
class BindingOverfitConfig:
    steps: int = 400
    lr: float = 5e-3
    num_traces: int = 32
    top_k: int = 16
    cue_dim: int = 16
    seed: int = 7

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.num_traces <= 0:
            raise ValueError("num_traces must be positive.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.cue_dim <= 0:
            raise ValueError("cue_dim must be positive.")


@dataclass(frozen=True)
class BindingOverfitGateResult:
    passed: bool
    failures: tuple[str, ...]
    full_report: BindingEdgeEvalReport
    no_binding_report: BindingEdgeEvalReport

    def to_log_fields(self) -> dict[str, str]:
        return {
            "passed": str(self.passed),
            "full_acc": f"{self.full_report.binding_edge_accuracy:.4f}",
            "full_pos_mass": f"{self.full_report.positive_binding_mass_mean:.4f}",
            "full_neg_mass": f"{self.full_report.negative_binding_mass_mean:.4f}",
            "no_binding_acc": f"{self.no_binding_report.binding_edge_accuracy:.4f}",
            "no_binding_pos_mass": f"{self.no_binding_report.positive_binding_mass_mean:.4f}",
            "no_binding_neg_mass": f"{self.no_binding_report.negative_binding_mass_mean:.4f}",
            "failures": ",".join(self.failures) if self.failures else "none",
        }


def default_overfit_examples() -> list[BindingEdgeExample]:
    """Small present-present edge set sized to overfit reliably with top_k=16."""

    return [
        BindingEdgeExample(
            text="alpha beta gamma",
            edges=(
                BindingEdge("alpha", "beta", 1),
                BindingEdge("beta", "gamma", 1),
                BindingEdge("alpha", "gamma", 0),
            ),
        ),
        BindingEdgeExample(
            text="north south east",
            edges=(
                BindingEdge("north", "south", 1),
                BindingEdge("south", "east", 1),
                BindingEdge("north", "east", 0),
            ),
        ),
    ]


def set_binding_layer_trainable(stack: BindingStack, *, trainable: bool) -> None:
    for parameter in stack.binding_layer.parameters():
        parameter.requires_grad = trainable


def train_binding_stack(
    examples: list[BindingEdgeExample],
    config: BindingOverfitConfig,
    *,
    train_binding: bool = True,
    trace: bool = False,
) -> BindingStack:
    stack = build_binding_stack(
        examples,
        cue_dim=config.cue_dim,
        num_traces=config.num_traces,
        top_k=config.top_k,
        seed=config.seed,
    )
    set_binding_layer_trainable(stack, trainable=train_binding)
    optimizer = torch.optim.Adam(
        (parameter for parameter in stack.parameters() if parameter.requires_grad),
        lr=config.lr,
    )

    for step in range(1, config.steps + 1):
        stack.train()
        optimizer.zero_grad()
        step_losses: list[Tensor] = []
        for example in examples:
            result = stack.forward_example(example)
            loss = stack.binding_edge_loss(result)
            if float(loss.detach()) > 0:
                step_losses.append(loss)
        if not step_losses:
            continue
        total = torch.stack(step_losses).mean()
        total.backward()
        optimizer.step()
        if trace and (step == 1 or step == config.steps or step % max(config.steps // 4, 1) == 0):
            report = evaluate_binding_edges(examples, stack=stack)
            LOGGER.info(
                "binding_overfit.train step=%s train_binding=%s loss=%.4f acc=%.4f pos_mass=%.4f neg_mass=%.4f",
                step,
                train_binding,
                float(total.detach()),
                report.binding_edge_accuracy,
                report.positive_binding_mass_mean,
                report.negative_binding_mass_mean,
            )

    stack.eval()
    return stack


def evaluate_overfit_gates(
    full_report: BindingEdgeEvalReport,
    no_binding_report: BindingEdgeEvalReport,
) -> BindingOverfitGateResult:
    failures: list[str] = []

    if full_report.binding_edge_accuracy < OVERFIT_MIN_ACCURACY:
        failures.append(
            f"full accuracy {full_report.binding_edge_accuracy:.3f} < {OVERFIT_MIN_ACCURACY:.2f}"
        )
    if full_report.positive_binding_mass_mean < OVERFIT_MIN_POSITIVE_MASS:
        failures.append(
            "full positive mass "
            f"{full_report.positive_binding_mass_mean:.3f} < {OVERFIT_MIN_POSITIVE_MASS:.2f}"
        )
    if full_report.negative_binding_mass_mean > OVERFIT_MAX_NEGATIVE_MASS:
        failures.append(
            "full negative mass "
            f"{full_report.negative_binding_mass_mean:.3f} > {OVERFIT_MAX_NEGATIVE_MASS:.2f}"
        )
    if full_report.binding_edge_accuracy <= no_binding_report.binding_edge_accuracy:
        failures.append(
            "full accuracy "
            f"{full_report.binding_edge_accuracy:.3f} "
            "must exceed no_binding "
            f"{no_binding_report.binding_edge_accuracy:.3f}"
        )

    return BindingOverfitGateResult(
        passed=not failures,
        failures=tuple(failures),
        full_report=full_report,
        no_binding_report=no_binding_report,
    )


def run_binding_overfit_sanity(
    examples: list[BindingEdgeExample] | None = None,
    config: BindingOverfitConfig | None = None,
    *,
    trace: bool = False,
) -> BindingOverfitGateResult:
    examples = default_overfit_examples() if examples is None else examples
    config = config or BindingOverfitConfig()

    if trace:
        LOGGER.info(
            "binding_overfit.start examples=%s steps=%s top_k=%s seed=%s",
            len(examples),
            config.steps,
            config.top_k,
            config.seed,
        )

    full_stack = train_binding_stack(examples, config, train_binding=True, trace=trace)
    no_binding_stack = train_binding_stack(examples, config, train_binding=False, trace=trace)

    full_report = evaluate_binding_edges(examples, stack=full_stack)
    no_binding_report = evaluate_binding_edges(examples, stack=no_binding_stack)
    result = evaluate_overfit_gates(full_report, no_binding_report)

    if trace:
        LOGGER.info("binding_overfit.done %s", " ".join(f"{k}={v}" for k, v in result.to_log_fields().items()))

    return result


def format_binding_overfit_report(result: BindingOverfitGateResult) -> str:
    lines = [
        "",
        "BINDING OVERFIT SANITY (C5)",
        f"passed={'yes' if result.passed else 'no'}",
        (
            f"full: acc={result.full_report.binding_edge_accuracy:.3f}  "
            f"pos_mass={result.full_report.positive_binding_mass_mean:.3f}  "
            f"neg_mass={result.full_report.negative_binding_mass_mean:.3f}"
        ),
        (
            f"no_binding: acc={result.no_binding_report.binding_edge_accuracy:.3f}  "
            f"pos_mass={result.no_binding_report.positive_binding_mass_mean:.3f}  "
            f"neg_mass={result.no_binding_report.negative_binding_mass_mean:.3f}"
        ),
        (
            f"gates: acc>={OVERFIT_MIN_ACCURACY:.2f}  "
            f"pos_mass>={OVERFIT_MIN_POSITIVE_MASS:.2f}  "
            f"neg_mass<={OVERFIT_MAX_NEGATIVE_MASS:.2f}  full>no_binding"
        ),
    ]
    if result.failures:
        lines.append("failures:")
        lines.extend(f"  - {failure}" for failure in result.failures)
    return "\n".join(lines)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run binding overfit sanity gates (Commit C5).")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num-traces", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if args.trace:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    result = run_binding_overfit_sanity(
        config=BindingOverfitConfig(
            steps=args.steps,
            lr=args.lr,
            num_traces=args.num_traces,
            top_k=args.top_k,
            cue_dim=args.cue_dim,
            seed=args.seed,
        ),
        trace=args.trace,
    )
    print(format_binding_overfit_report(result))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
