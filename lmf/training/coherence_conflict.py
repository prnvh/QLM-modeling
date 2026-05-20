"""Coherence / conflict training hook (Commit F2).

Light-weight hook: compatible continuation pairs should yield higher
``coexistence - conflict`` interference scores than incompatible pairs.
Weight stays small until basins stabilize.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.field.loop import FieldLoop, FieldLoopConfig  # noqa: E402
from lmf.core.dmf.trace_router import route_text  # noqa: E402
from lmf.core.state.types import InterferenceState  # noqa: E402
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "stage1" / "coherence_conflict.jsonl"


@dataclass(frozen=True)
class CoherenceConflictExample:
    text_a: str
    text_b: str
    label: int


@dataclass
class CoherenceConflictConfig:
    loss_weight: float = 0.01
    margin: float = 0.10
    num_traces: int = 64
    top_k: int = 12
    cue_dim: int = 16
    num_basins: int = 16
    settling_steps: int = 2
    seed: int = 7

    def __post_init__(self) -> None:
        if self.loss_weight < 0.0:
            raise ValueError("loss_weight must be non-negative.")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative.")


@dataclass(frozen=True)
class CoherenceConflictMetrics:
    loss: float
    compatible_score_mean: float
    incompatible_score_mean: float
    margin_violations: int
    num_pairs: int


def load_coherence_conflict_examples(path: Path | str) -> list[CoherenceConflictExample]:
    examples: list[CoherenceConflictExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1. Got {label}.")
        examples.append(
            CoherenceConflictExample(
                text_a=str(row["text_a"]),
                text_b=str(row["text_b"]),
                label=label,
            )
        )
    if not examples:
        raise ValueError("coherence conflict dataset is empty.")
    return examples


def interference_coherence_score(interference_state: InterferenceState) -> Tensor:
    """Higher when coexistence dominates conflict."""

    if interference_state.coexistence_score is None or interference_state.conflict_score is None:
        raise ValueError("interference_state must include coexistence_score and conflict_score.")
    return (
        interference_state.coexistence_score.squeeze(-1)
        - interference_state.conflict_score.squeeze(-1)
    )


def coherence_conflict_margin_loss(
    *,
    compatible_scores: Tensor,
    incompatible_scores: Tensor,
    margin: float,
) -> Tensor:
    if compatible_scores.numel() == 0 or incompatible_scores.numel() == 0:
        device = compatible_scores.device if compatible_scores.numel() else incompatible_scores.device
        return torch.zeros((), device=device)
    return F.relu(margin + incompatible_scores.mean() - compatible_scores.mean())


class CoherenceConflictHook(nn.Module):
    """Score continuation pairs using basin-competition interference diagnostics."""

    def __init__(self, config: CoherenceConflictConfig) -> None:
        super().__init__()
        self.config = config
        self.field_loop = FieldLoop(
            FieldLoopConfig(
                cue_dim=config.cue_dim,
                content_dim=config.cue_dim,
                active_traces=config.top_k,
                num_basins=config.num_basins,
                settling_steps=config.settling_steps,
            )
        )

    def score_text(self, text: str) -> Tensor:
        cue_packet, _routing, active_region = route_text(
            text,
            num_traces=self.config.num_traces,
            top_k=self.config.top_k,
            cue_dim=self.config.cue_dim,
            seed=self.config.seed,
        )
        basin_state = self.field_loop.make_basin_state(
            active_region.trace_amp.shape[0],
            device=active_region.trace_amp.device,
        )
        output = self.field_loop(cue_packet, active_region, basin_state)
        if not isinstance(output.interference_state, InterferenceState):
            raise TypeError("field loop must return InterferenceState.")
        return interference_coherence_score(output.interference_state)

    def score_pair(self, text_a: str, text_b: str) -> Tensor:
        combined = f"{text_a} {text_b}"
        return self.score_text(combined)

    def loss_from_examples(self, examples: list[CoherenceConflictExample]) -> tuple[Tensor, CoherenceConflictMetrics]:
        compatible: list[Tensor] = []
        incompatible: list[Tensor] = []
        violations = 0

        for example in examples:
            score = self.score_pair(example.text_a, example.text_b)
            if example.label == 1:
                compatible.append(score)
            else:
                incompatible.append(score)
                if float(score.item()) > 0.0:
                    violations += 1

        if not compatible or not incompatible:
            raise ValueError("dataset must contain both compatible and incompatible pairs.")

        compatible_tensor = torch.stack(compatible)
        incompatible_tensor = torch.stack(incompatible)
        raw_loss = coherence_conflict_margin_loss(
            compatible_scores=compatible_tensor,
            incompatible_scores=incompatible_tensor,
            margin=self.config.margin,
        )
        weighted = raw_loss * self.config.loss_weight
        metrics = CoherenceConflictMetrics(
            loss=float(weighted.detach().item()),
            compatible_score_mean=float(compatible_tensor.mean().detach().item()),
            incompatible_score_mean=float(incompatible_tensor.mean().detach().item()),
            margin_violations=violations,
            num_pairs=len(examples),
        )
        return weighted, metrics


def evaluate_coherence_conflict(
    hook: CoherenceConflictHook,
    examples: list[CoherenceConflictExample],
) -> CoherenceConflictMetrics:
    hook.eval()
    with torch.no_grad():
        _loss, metrics = hook.loss_from_examples(examples)
    return metrics


def format_coherence_conflict_report(metrics: CoherenceConflictMetrics) -> str:
    return (
        f"loss={metrics.loss:.4f} | compatible_mean={metrics.compatible_score_mean:.4f} | "
        f"incompatible_mean={metrics.incompatible_score_mean:.4f} | "
        f"margin_violations={metrics.margin_violations}/{metrics.num_pairs}"
    )


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coherence/conflict training hook (Commit F2).")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--loss-weight", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    examples = load_coherence_conflict_examples(args.data_path)
    hook = CoherenceConflictHook(
        CoherenceConflictConfig(loss_weight=args.loss_weight, margin=args.margin)
    )
    metrics = evaluate_coherence_conflict(hook, examples)
    _safe_print(format_coherence_conflict_report(metrics))
    if args.trace:
        LOGGER.info("coherence_conflict_eval | %s", format_coherence_conflict_report(metrics))


__all__ = [
    "CoherenceConflictConfig",
    "CoherenceConflictExample",
    "CoherenceConflictHook",
    "CoherenceConflictMetrics",
    "coherence_conflict_margin_loss",
    "evaluate_coherence_conflict",
    "format_coherence_conflict_report",
    "interference_coherence_score",
    "load_coherence_conflict_examples",
]
