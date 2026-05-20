"""Cue-pair binding supervision through the full binding stack (Commit C3)."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.training.binding_edges import (  # noqa: E402
    BindingEdgeExample,
    load_binding_edges,
    resolve_example_edges,
)
from lmf.training.binding_stack import (  # noqa: E402
    BindingForwardResult,
    BindingStack,
    build_binding_stack,
    best_pair_mass,
    trace_indices_for_positions,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_BINDING_EDGES_PATH = PROJECT_ROOT / "data" / "stage1" / "binding_edges.jsonl"
DEFAULT_DECISION_THRESHOLD = 0.5


def _log_trace(logger: logging.Logger, enabled: bool, event: str, **fields: object) -> None:
    if not enabled:
        return
    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s%s", event, f" | {details}" if details else "")


def topk_edge_mass(result: BindingForwardResult, *, indices_a: list[int], indices_b: list[int]) -> float:
    if not indices_a or not indices_b:
        return 0.0
    edge_index = result.binding_state.edge_index[0]
    strengths = result.binding_state.relation_strength[0]
    best = 0.0
    for edge in range(edge_index.shape[1]):
        src = int(edge_index[0, edge].item())
        dst = int(edge_index[1, edge].item())
        if (src in indices_a and dst in indices_b) or (src in indices_b and dst in indices_a):
            best = max(best, float(strengths[edge].item()))
    return max(best, 0.0)


@dataclass
class BindingEdgeRecord:
    cue_a: str
    cue_b: str
    label: int
    pair_mass: float
    topk_mass: float
    predicted: int
    covered: bool
    active: bool
    cue_a_positions: tuple[int, ...] = ()
    cue_b_positions: tuple[int, ...] = ()


@dataclass
class BindingEdgeEvalReport:
    binding_edge_loss: float
    binding_edge_accuracy: float
    positive_binding_mass_mean: float
    negative_binding_mass_mean: float
    positive_edge_coverage: float
    negative_edge_coverage: float
    positive_active_coverage: float
    negative_active_coverage: float
    num_edges: int
    records: list[BindingEdgeRecord] = field(default_factory=list)

    def to_log_fields(self) -> dict[str, str]:
        return {
            "binding_edge_loss": f"{self.binding_edge_loss:.4f}",
            "binding_edge_accuracy": f"{self.binding_edge_accuracy:.4f}",
            "positive_binding_mass_mean": f"{self.positive_binding_mass_mean:.4f}",
            "negative_binding_mass_mean": f"{self.negative_binding_mass_mean:.4f}",
            "positive_edge_coverage": f"{self.positive_edge_coverage:.4f}",
            "negative_edge_coverage": f"{self.negative_edge_coverage:.4f}",
            "positive_active_coverage": f"{self.positive_active_coverage:.4f}",
            "negative_active_coverage": f"{self.negative_active_coverage:.4f}",
            "num_edges": str(self.num_edges),
        }


def _records_from_result(
    result: BindingForwardResult,
    *,
    decision_threshold: float,
) -> list[BindingEdgeRecord]:
    records: list[BindingEdgeRecord] = []
    for edge in result.resolved_edges:
        indices_a = trace_indices_for_positions(result.active_region, edge.cue_a_positions)
        indices_b = trace_indices_for_positions(result.active_region, edge.cue_b_positions)
        covered = bool(indices_a and indices_b)
        mass_tensor = best_pair_mass(result.pair_mass, indices_a=indices_a, indices_b=indices_b)
        mass = float(mass_tensor.detach().item())
        topk = topk_edge_mass(result, indices_a=indices_a, indices_b=indices_b)
        predicted = int(mass >= decision_threshold)
        records.append(
            BindingEdgeRecord(
                cue_a=edge.cue_a,
                cue_b=edge.cue_b,
                label=edge.label,
                pair_mass=mass,
                topk_mass=topk,
                predicted=predicted,
                covered=covered,
                active=bool(
                    covered
                    and (
                        (edge.label == 1 and mass >= decision_threshold)
                        or (edge.label == 0 and mass < decision_threshold)
                    )
                ),
                cue_a_positions=edge.cue_a_positions,
                cue_b_positions=edge.cue_b_positions,
            )
        )
    return records


def _summarize_records(records: list[BindingEdgeRecord], *, loss: float) -> BindingEdgeEvalReport:
    if not records:
        return BindingEdgeEvalReport(
            binding_edge_loss=loss,
            binding_edge_accuracy=0.0,
            positive_binding_mass_mean=0.0,
            negative_binding_mass_mean=0.0,
            positive_edge_coverage=0.0,
            negative_edge_coverage=0.0,
            positive_active_coverage=0.0,
            negative_active_coverage=0.0,
            num_edges=0,
        )

    positives = [record for record in records if record.label == 1]
    negatives = [record for record in records if record.label == 0]

    def _mean_mass(items: list[BindingEdgeRecord]) -> float:
        return sum(record.pair_mass for record in items) / len(items) if items else 0.0

    def _coverage(items: list[BindingEdgeRecord]) -> float:
        return sum(1 for record in items if record.covered) / len(items) if items else 0.0

    def _active_coverage(items: list[BindingEdgeRecord]) -> float:
        return sum(1 for record in items if record.active) / len(items) if items else 0.0

    accuracy = sum(1 for record in records if record.predicted == record.label) / len(records)
    return BindingEdgeEvalReport(
        binding_edge_loss=loss,
        binding_edge_accuracy=accuracy,
        positive_binding_mass_mean=_mean_mass(positives),
        negative_binding_mass_mean=_mean_mass(negatives),
        positive_edge_coverage=_coverage(positives),
        negative_edge_coverage=_coverage(negatives),
        positive_active_coverage=_active_coverage(positives),
        negative_active_coverage=_active_coverage(negatives),
        num_edges=len(records),
        records=records,
    )


def evaluate_binding_edges_on_example(
    example: BindingEdgeExample,
    *,
    stack: BindingStack,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    trace: bool = False,
    logger: logging.Logger = LOGGER,
) -> BindingEdgeEvalReport:
    stack.eval()
    with torch.no_grad():
        result = stack.forward_example(example)
    loss = float(stack.binding_edge_loss(result).detach().item())
    records = _records_from_result(result, decision_threshold=decision_threshold)
    report = _summarize_records(records, loss=loss)

    _log_trace(
        logger,
        trace,
        "binding_edge_eval.example",
        text=example.text,
        accuracy=f"{report.binding_edge_accuracy:.3f}",
        loss=f"{report.binding_edge_loss:.3f}",
    )
    return report


def evaluate_binding_edges(
    examples: list[BindingEdgeExample],
    *,
    stack: BindingStack,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    trace: bool = False,
) -> BindingEdgeEvalReport:
    all_records: list[BindingEdgeRecord] = []
    losses: list[float] = []
    for example in examples:
        report = evaluate_binding_edges_on_example(
            example,
            stack=stack,
            decision_threshold=decision_threshold,
            trace=trace,
        )
        all_records.extend(report.records)
        losses.append(report.binding_edge_loss)
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    return _summarize_records(all_records, loss=mean_loss)


def format_binding_edge_report(*, text: str, report: BindingEdgeEvalReport) -> str:
    lines = [
        "",
        "BINDING EDGE EVAL",
        text,
        (
            f"loss={report.binding_edge_loss:.3f}  acc={report.binding_edge_accuracy:.3f}  "
            f"pos_mass={report.positive_binding_mass_mean:.3f}  neg_mass={report.negative_binding_mass_mean:.3f}"
        ),
        (
            f"pos_cov={report.positive_edge_coverage:.3f}  neg_cov={report.negative_edge_coverage:.3f}  "
            f"pos_active={report.positive_active_coverage:.3f}  neg_active={report.negative_active_coverage:.3f}"
        ),
        "",
        f"{'cue_a':<10} {'cue_b':<10} {'label':>5} {'mass':>6} {'topk':>6} {'pos_a':<8} {'pos_b':<8} {'ok':>4}",
        "-" * 70,
    ]
    for record in report.records:
        ok = "yes" if record.predicted == record.label else "no"
        if not record.covered:
            ok = "miss"
        pos_a = ",".join(str(value) for value in record.cue_a_positions)
        pos_b = ",".join(str(value) for value in record.cue_b_positions)
        lines.append(
            f"{record.cue_a:<10} {record.cue_b:<10} {record.label:>5} "
            f"{record.pair_mass:>6.3f} {record.topk_mass:>6.3f} {pos_a:<8} {pos_b:<8} {ok:>4}"
        )
    return "\n".join(lines)


def find_example_for_text(examples: list[BindingEdgeExample], text: str) -> BindingEdgeExample:
    normalized = text.strip()
    for example in examples:
        if example.text == normalized:
            return example
    raise ValueError(f"no binding edge example found for text: {normalized!r}")


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding))


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate binding through the full binding stack.")
    parser.add_argument("text", nargs="*", help="Optional one prompt from the edges file.")
    parser.add_argument("--edges-file", type=str, default=str(DEFAULT_BINDING_EDGES_PATH))
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=DEFAULT_DECISION_THRESHOLD)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    stack = build_binding_stack(
        load_binding_edges(args.edges_file),
        cue_dim=args.cue_dim,
        num_traces=args.num_traces,
        top_k=args.top_k,
        seed=args.seed,
    )

    if args.text:
        parts = args.text[1:] if args.text[0].lower() == "text" else args.text
        text = " ".join(parts)
        example = find_example_for_text(load_binding_edges(args.edges_file), text)
        report = evaluate_binding_edges_on_example(
            example,
            stack=stack,
            decision_threshold=args.threshold,
            trace=args.trace,
        )
        _safe_print(format_binding_edge_report(text=example.text, report=report))
        return

    examples = load_binding_edges(args.edges_file)
    aggregate = evaluate_binding_edges(
        examples,
        stack=stack,
        decision_threshold=args.threshold,
        trace=args.trace,
    )
    _safe_print("DATASET " + "  ".join(f"{k}={v}" for k, v in aggregate.to_log_fields().items()))
    for example in examples:
        report = evaluate_binding_edges_on_example(example, stack=stack, decision_threshold=args.threshold)
        _safe_print(format_binding_edge_report(text=example.text, report=report))


__all__ = [
    "BindingEdgeEvalReport",
    "BindingEdgeRecord",
    "evaluate_binding_edges",
    "evaluate_binding_edges_on_example",
    "format_binding_edge_report",
]


if __name__ == "__main__":
    main()
