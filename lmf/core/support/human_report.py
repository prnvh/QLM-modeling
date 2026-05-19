"""Human-readable reports for manual pipeline inspection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lmf.core.field.types import FieldLoopOutput
from lmf.core.state.types import BindingState, InterferenceState


def _strength_word(value: float, *, strong: float, medium: float) -> str:
    if value >= strong:
        return "strong"
    if value >= medium:
        return "medium"
    return "weak"


@dataclass
class PipelineInspection:
    """Everything needed to print a readable end-to-end report."""

    input_text: str
    tokens: list[str]
    token_rows: list[tuple[int, int, str]]
    routing_mode: str
    score_mode: str
    num_traces: int
    top_k: int
    trace_ids: list[int]
    match_scores: list[float]
    activations_before: list[float]
    cue_drives: list[float]
    field_output: FieldLoopOutput | None = None
    activations_after: list[float] | None = None


def format_header(title: str) -> str:
    line = "=" * len(title)
    return f"\n{title}\n{line}\n"


def format_token_section(*, input_text: str, tokens: list[str], token_rows: list[tuple[int, int, str]]) -> str:
    lines = [
        format_header("1) TEXT IN"),
        f"Sentence: {input_text}",
        f"Tokens ({len(tokens)}): {' | '.join(tokens)}",
        "",
        "Token list:",
    ]
    for position, token_id, token in token_rows:
        lines.append(f"  {position:>2}. {token!r:16}  id={token_id}")
    return "\n".join(lines)


def format_routing_section(inspection: PipelineInspection) -> str:
    ids = inspection.trace_ids
    scores = inspection.match_scores
    amps = inspection.activations_before
    drives = inspection.cue_drives

    inactive = inspection.num_traces - len(ids)
    max_score = max(scores) if scores else 1.0
    strong_cut = max_score * 0.66
    medium_cut = max_score * 0.33

    lines = [
        format_header("2) SPARSE ROUTING (which memory slots woke up)"),
        (
            f"Compared your cues to {inspection.num_traces} learnable trace slots, "
            f"kept the best {inspection.top_k}, left {inactive} inactive."
        ),
        f"Mode: {inspection.routing_mode} cues, {inspection.score_mode} similarity.",
        "",
        "NOTE: Trace ids are slot numbers, not English words. Training shapes what each slot means.",
        "",
        f"{'Rank':<5} {'Trace':<10} {'Match':<8} {'Activation':<12} {'Cue push':<10} {'Match level':<12}",
        "-" * 62,
    ]

    for rank, trace_id, score, amp, drive in zip(
        range(1, len(ids) + 1),
        ids,
        scores,
        amps,
        drives,
        strict=True,
    ):
        lines.append(
            f"{rank:<5} trace_{trace_id:<4} {score:>7.3f} {amp:>11.3f} {drive:>9.3f} "
            f"{_strength_word(score, strong=strong_cut, medium=medium_cut):<12}"
        )

    lines.extend(
        [
            "",
            "How to read this:",
            "  Match      = cue matched this slot's key (higher = better fit right now).",
            "  Activation = how 'on' the slot is before field settling.",
            "  Cue push   = how hard the input is pressing this slot.",
        ]
    )
    return "\n".join(lines)


def format_binding_section(binding_state: BindingState, trace_ids: list[int]) -> str:
    edge_index = binding_state.edge_index[0].detach().cpu()
    strengths = binding_state.relation_strength[0].detach().cpu()

    pairs: list[tuple[int, int, float]] = []
    for edge in range(edge_index.shape[1]):
        strength = float(strengths[edge].item())
        if strength <= 1e-4 or not torch.isfinite(torch.tensor(strength)):
            continue
        src = int(edge_index[0, edge].item())
        dst = int(edge_index[1, edge].item())
        pairs.append((src, dst, strength))

    pairs.sort(key=lambda item: item[2], reverse=True)
    top_pairs = pairs[:12]

    id_to_slot = {local: trace_ids[local] for local in range(len(trace_ids))}

    lines = [
        format_header("3) BINDING (which active slots are linked)"),
        "Soft edges between active slots. This is NOT copying words between slots.",
        f"Showing top {len(top_pairs)} edges by strength.",
        "",
        f"{'From':<12} {'To':<12} {'Strength':<10} {'Plain English':<30}",
        "-" * 68,
    ]

    if not top_pairs:
        lines.append(
            "(no strong binding edges yet — expected with random/untrained weights; "
            "edges will sharpen after binding training in later commits)"
        )
    else:
        for src, dst, strength in top_pairs:
            from_trace = id_to_slot.get(src, src)
            to_trace = id_to_slot.get(dst, dst)
            phrase = _binding_phrase(strength)
            lines.append(
                f"trace_{from_trace:<5} trace_{to_trace:<5} {strength:>8.3f} {phrase:<30}"
            )

    return "\n".join(lines)


def _binding_phrase(strength: float) -> str:
    if strength >= 0.5:
        return "tight link"
    if strength >= 0.2:
        return "moderate link"
    return "weak link"


def format_settling_section(
    *,
    trace_ids: list[int],
    before: list[float],
    after: list[float],
    steps_run: int,
    interference_state: InterferenceState | None,
) -> str:
    lines = [
        format_header("4) FIELD SETTLING (how strengths changed)"),
        f"Ran {steps_run} settling steps on the active slots only.",
        "",
        f"{'Trace':<10} {'Before':<10} {'After':<10} {'Change':<10} {'Trend':<10}",
        "-" * 52,
    ]

    for trace_id, start, end in zip(trace_ids, before, after, strict=True):
        delta = end - start
        if abs(delta) < 0.01:
            trend = "stable"
        elif delta > 0:
            trend = "rose"
        else:
            trend = "fell"
        lines.append(
            f"trace_{trace_id:<4} {start:>9.3f} {end:>9.3f} {delta:>+9.3f} {trend:<10}"
        )

    if interference_state is not None:
        pair_energy = interference_state.pair_energy
        contradiction = interference_state.contradiction
        if pair_energy is not None and contradiction is not None:
            pe = float(pair_energy[0].item())
            ct = float(contradiction[0].item())
            lines.extend(
                [
                    "",
                    "Interference (binding-gated):",
                    f"  compatibility pressure: {pe:.3f}",
                    f"  contradiction signal: {ct:.3f}",
                ]
            )

    lines.extend(
        [
            "",
            "How to read this:",
            "  Before/After = slot activation strength. Settling nudges the active field, not the whole bank.",
        ]
    )
    return "\n".join(lines)


def format_pipeline_report(inspection: PipelineInspection) -> str:
    """Build the full readable report."""

    sections = [
        format_token_section(
            input_text=inspection.input_text,
            tokens=inspection.tokens,
            token_rows=inspection.token_rows,
        ),
        format_routing_section(inspection),
    ]

    if inspection.field_output is not None and inspection.field_output.binding_state is not None:
        binding_state = inspection.field_output.binding_state
        if isinstance(binding_state, BindingState):
            sections.append(format_binding_section(binding_state, inspection.trace_ids))

    if (
        inspection.field_output is not None
        and inspection.activations_after is not None
        and inspection.activations_before is not None
    ):
        sections.append(
            format_settling_section(
                trace_ids=inspection.trace_ids,
                before=inspection.activations_before,
                after=inspection.activations_after,
                steps_run=inspection.field_output.steps_run,
                interference_state=inspection.field_output.interference_state
                if isinstance(inspection.field_output.interference_state, InterferenceState)
                else None,
            )
        )

    sections.append(
        format_header("Done")
        + "This is inspection of the current weights (seed-controlled), not a trained model yet."
    )
    return "\n".join(sections)


def tensors_to_ranked_lists(
    trace_ids: Tensor,
    scores: Tensor,
    trace_amp: Tensor,
    cue_drive: Tensor,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Sort active traces by match score descending for readable output."""

    ids = trace_ids[0].detach().cpu().tolist()
    score_values = scores[0].detach().cpu().tolist()
    amp_values = trace_amp[0].detach().cpu().tolist()
    drive_values = cue_drive[0].detach().cpu().tolist()

    ranked = sorted(
        zip(ids, score_values, amp_values, drive_values),
        key=lambda row: row[1],
        reverse=True,
    )
    if not ranked:
        return [], [], [], []

    ids_out, scores_out, amps_out, drives_out = zip(*ranked)
    return list(ids_out), list(scores_out), list(amps_out), list(drives_out)


__all__ = [
    "PipelineInspection",
    "format_pipeline_report",
    "format_routing_section",
    "tensors_to_ranked_lists",
]
