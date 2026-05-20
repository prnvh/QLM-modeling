"""Same-checkpoint cognitive-state channel ablation evaluation (Commit E3)."""

from __future__ import annotations

import argparse
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

from lmf.core.decode.decoder import (  # noqa: E402
    STANDARD_DECODER_ABLATIONS,
    ChannelAblation,
    CognitiveStateDecoder,
    DecoderConfig,
)
from lmf.core.dmf.trace_router import route_text  # noqa: E402
from lmf.core.field.context_op import ContextOp, ContextOpConfig  # noqa: E402
from lmf.core.field.loop import FieldLoop, FieldLoopConfig  # noqa: E402
from lmf.core.state.cognitive_state import build_cognitive_state_from_field_loop  # noqa: E402
from lmf.core.state.types import CognitiveState  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AblationReadoutRecord:
    name: str
    readout: Tensor
    cosine_to_full: float | None


@dataclass(frozen=True)
class DecoderAblationReport:
    text: str
    settling_steps: int
    records: tuple[AblationReadoutRecord, ...]

    @property
    def full_readout(self) -> Tensor:
        for record in self.records:
            if record.name == "full":
                return record.readout
        raise ValueError("report is missing a full-state readout.")


def cosine_distance(a: Tensor, b: Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError("cosine_distance requires matching tensor shapes.")
    a_norm = F.normalize(a, dim=-1, eps=1e-6)
    b_norm = F.normalize(b, dim=-1, eps=1e-6)
    return float((1.0 - (a_norm * b_norm).sum(dim=-1)).mean().item())


def evaluate_decoder_ablations(
    state: CognitiveState,
    decoder: CognitiveStateDecoder,
    *,
    ablations: dict[str, ChannelAblation] | None = None,
) -> tuple[AblationReadoutRecord, ...]:
    """Run same-checkpoint channel ablations on one cognitive state."""

    presets = ablations or STANDARD_DECODER_ABLATIONS
    decoder.eval()
    records: list[AblationReadoutRecord] = []
    full_readout: Tensor | None = None

    with torch.no_grad():
        for name, ablation in presets.items():
            output = decoder(state, ablation=ablation)
            if name == "full":
                full_readout = output.readout
                records.append(AblationReadoutRecord(name=name, readout=output.readout, cosine_to_full=0.0))
                continue
            if full_readout is None:
                raise RuntimeError("full ablation must be evaluated first.")
            delta = cosine_distance(full_readout, output.readout)
            records.append(AblationReadoutRecord(name=name, readout=output.readout, cosine_to_full=delta))

    return tuple(records)


def evaluate_decoder_ablations_on_text(
    text: str,
    *,
    decoder: CognitiveStateDecoder,
    field_loop: FieldLoop,
    context_op: ContextOp,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    settling_steps: int = 3,
    seed: int = 7,
    ablations: dict[str, ChannelAblation] | None = None,
) -> DecoderAblationReport:
    cue_packet, _routing, active_region = route_text(
        text,
        num_traces=num_traces,
        top_k=top_k,
        cue_dim=cue_dim,
        seed=seed,
    )
    basin_state = field_loop.make_basin_state(
        active_region.trace_amp.shape[0],
        device=active_region.trace_amp.device,
    )
    field_loop.eval()
    context_op.eval()

    with torch.no_grad():
        context = context_op(cue_packet, active_region)
        field_output = field_loop(
            cue_packet,
            active_region,
            basin_state,
            steps=settling_steps,
        )
        state = build_cognitive_state_from_field_loop(
            cue_packet=cue_packet,
            active_region=active_region,
            basin_state=basin_state,
            field_output=field_output,
            context_summary=context.context_summary,
        )

    records = evaluate_decoder_ablations(state, decoder, ablations=ablations)
    return DecoderAblationReport(text=text, settling_steps=settling_steps, records=records)


def evaluate_settling_k_ablation(
    text: str,
    *,
    decoder: CognitiveStateDecoder,
    field_loop: FieldLoop,
    context_op: ContextOp,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    full_steps: int = 3,
    reduced_steps: int = 1,
    seed: int = 7,
) -> tuple[AblationReadoutRecord, AblationReadoutRecord]:
    """Compare K=full vs K=1 settling from the same checkpoint."""

    full_report = evaluate_decoder_ablations_on_text(
        text,
        decoder=decoder,
        field_loop=field_loop,
        context_op=context_op,
        num_traces=num_traces,
        top_k=top_k,
        cue_dim=cue_dim,
        settling_steps=full_steps,
        seed=seed,
        ablations={"full": STANDARD_DECODER_ABLATIONS["full"]},
    )
    reduced_report = evaluate_decoder_ablations_on_text(
        text,
        decoder=decoder,
        field_loop=field_loop,
        context_op=context_op,
        num_traces=num_traces,
        top_k=top_k,
        cue_dim=cue_dim,
        settling_steps=reduced_steps,
        seed=seed,
        ablations={"full": STANDARD_DECODER_ABLATIONS["full"]},
    )
    full_record = full_report.records[0]
    reduced_record = reduced_report.records[0]
    delta = cosine_distance(full_record.readout, reduced_record.readout)
    return (
        AblationReadoutRecord(name=f"K={full_steps}", readout=full_record.readout, cosine_to_full=0.0),
        AblationReadoutRecord(name=f"K={reduced_steps}", readout=reduced_record.readout, cosine_to_full=delta),
    )


def format_decoder_ablation_report(report: DecoderAblationReport) -> str:
    lines = [
        f"text: {report.text}",
        f"settling_steps: {report.settling_steps}",
        "name | cosine_to_full",
    ]
    for record in report.records:
        delta = "0.000" if record.cosine_to_full is None else f"{record.cosine_to_full:.3f}"
        lines.append(f"{record.name} | {delta}")
    return "\n".join(lines)


def build_decoder_eval_stack(
    *,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    num_basins: int = 32,
    settling_steps: int = 3,
    readout_dim: int = 16,
) -> tuple[FieldLoop, ContextOp, CognitiveStateDecoder]:
    field_loop = FieldLoop(
        FieldLoopConfig(
            cue_dim=cue_dim,
            content_dim=cue_dim,
            active_traces=top_k,
            num_basins=num_basins,
            settling_steps=settling_steps,
        )
    )
    context_op = ContextOp(
        ContextOpConfig(
            cue_dim=cue_dim,
            active_traces=top_k,
            num_basins=num_basins,
        )
    )
    decoder = CognitiveStateDecoder(
        DecoderConfig(content_dim=cue_dim, readout_dim=readout_dim)
    )
    return field_loop, context_op, decoder


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate same-checkpoint decoder channel ablations.")
    parser.add_argument(
        "text",
        nargs="+",
        help='Text to process. You may optionally start with the word "text".',
    )
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--num-basins", type=int, default=32)
    parser.add_argument("--settling-steps", type=int, default=3)
    parser.add_argument("--readout-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)

    field_loop, context_op, decoder = build_decoder_eval_stack(
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        num_basins=args.num_basins,
        settling_steps=args.settling_steps,
        readout_dim=args.readout_dim,
    )
    report = evaluate_decoder_ablations_on_text(
        text,
        decoder=decoder,
        field_loop=field_loop,
        context_op=context_op,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        settling_steps=args.settling_steps,
        seed=args.seed,
    )
    k_full, k_one = evaluate_settling_k_ablation(
        text,
        decoder=decoder,
        field_loop=field_loop,
        context_op=context_op,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        full_steps=args.settling_steps,
        reduced_steps=1,
        seed=args.seed,
    )

    _safe_print(format_decoder_ablation_report(report))
    _safe_print("")
    _safe_print("settling ablation (same checkpoint):")
    _safe_print(f"{k_full.name} | cosine_to_full=0.000")
    _safe_print(f"{k_one.name} | cosine_to_full={k_one.cosine_to_full:.3f}")


__all__ = [
    "AblationReadoutRecord",
    "DecoderAblationReport",
    "build_decoder_eval_stack",
    "cosine_distance",
    "evaluate_decoder_ablations",
    "evaluate_decoder_ablations_on_text",
    "evaluate_settling_k_ablation",
    "format_decoder_ablation_report",
]


if __name__ == "__main__":
    main()
