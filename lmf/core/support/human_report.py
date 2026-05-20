"""Human-readable reports for manual pipeline inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lmf.core.field.types import FieldLoopOutput
from lmf.core.state.types import BindingState, InterferenceState
from lmf.core.dmf.cue_provenance import format_cue_type_label

if TYPE_CHECKING:
    from lmf.core.decode.readout_channels import ReadoutChannelInspection


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
    source_cue_ids: list[int] | None = None
    source_token_ids: list[int] | None = None
    source_tokens: list[str] | None = None
    source_spans: list[tuple[int, int]] | None = None
    cue_types: list[str] | None = None
    normalized_cue_ids: list[int] | None = None
    field_output: FieldLoopOutput | None = None
    activations_after: list[float] | None = None
    readout_inspection: ReadoutChannelInspection | None = None


def format_header(title: str) -> str:
    return f"\n{title}\n"


def resolve_source_tokens(
    token_rows: list[tuple[int, int, str]],
    source_cue_ids: list[int] | None,
    cue_types: list[str] | None = None,
) -> list[str] | None:
    """Map routing cue positions to the tokenizer token that drove each trace."""

    if source_cue_ids is None:
        return None

    pos_to_token = {pos: token for pos, _token_id, token in token_rows}
    resolved: list[str] = []
    cue_type_list = cue_types or ["token"] * len(source_cue_ids)
    for cue_id, cue_type in zip(source_cue_ids, cue_type_list, strict=True):
        if cue_id < 0 or cue_type == "pooled":
            resolved.append("(pooled)")
        elif cue_id in pos_to_token:
            resolved.append(pos_to_token[cue_id])
        else:
            resolved.append("?")
    return resolved


def format_token_section(*, input_text: str, tokens: list[str], token_rows: list[tuple[int, int, str]]) -> str:
    _ = tokens
    rows = "  ".join(f"{pos}:{token}" for pos, _token_id, token in token_rows)
    return "\n".join(
        [
            format_header("1) TOKENS"),
            input_text,
            rows,
        ]
    )


def format_routing_section(inspection: PipelineInspection) -> str:
    ids = inspection.trace_ids
    scores = inspection.match_scores
    amps = inspection.activations_before

    inactive = inspection.num_traces - len(ids)
    has_tokens = inspection.source_tokens is not None

    lines = [
        format_header(f"2) ROUTING  {len(ids)}/{inspection.num_traces} active, {inspection.routing_mode}/{inspection.score_mode}"),
    ]

    if has_tokens:
        lines.extend(
            [
                f"{'#':<3} {'trace':<8} {'match':>6} {'act':>5}  from token",
                "-" * 40,
            ]
        )
    else:
        lines.extend(
            [
                f"{'#':<3} {'trace':<8} {'match':>6} {'act':>5}",
                "-" * 28,
            ]
        )

    for rank, trace_id, score, amp in zip(
        range(1, len(ids) + 1),
        ids,
        scores,
        amps,
        strict=True,
    ):
        if has_tokens:
            token = inspection.source_tokens[rank - 1]  # type: ignore[index]
            lines.append(f"{rank:<3} trace_{trace_id:<3} {score:>6.3f} {amp:>5.3f}  {token!r}")
        else:
            lines.append(f"{rank:<3} trace_{trace_id:<3} {score:>6.3f} {amp:>5.3f}")

    if has_tokens:
        driven = sorted(
            {
                token
                for token in inspection.source_tokens  # type: ignore[union-attr]
                if token not in {"(pooled)", "?"}
            }
        )
        if driven:
            lines.append(f"tokens used: {', '.join(driven)}")

    if inactive:
        lines.append(f"{inactive} trace slots inactive")
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
        format_header("3) BINDING"),
        f"{'from':<10} {'to':<10} {'strength':>8}",
        "-" * 32,
    ]

    if not top_pairs:
        lines.append("(none)")
    else:
        for src, dst, strength in top_pairs:
            from_trace = id_to_slot.get(src, src)
            to_trace = id_to_slot.get(dst, dst)
            lines.append(f"trace_{from_trace:<4} trace_{to_trace:<4} {strength:>8.3f}")

    return "\n".join(lines)


def format_settling_section(
    *,
    trace_ids: list[int],
    before: list[float],
    after: list[float],
    steps_run: int,
    interference_state: InterferenceState | None,
) -> str:
    lines = [
        format_header(f"4) SETTLING  {steps_run} steps"),
        f"{'trace':<8} {'before':>7} {'after':>7} {'delta':>7}",
        "-" * 32,
    ]

    for trace_id, start, end in zip(trace_ids, before, after, strict=True):
        delta = end - start
        lines.append(f"trace_{trace_id:<3} {start:>7.3f} {end:>7.3f} {delta:>+7.3f}")

    if interference_state is not None:
        pair_energy = interference_state.pair_energy
        contradiction = interference_state.contradiction
        if pair_energy is not None and contradiction is not None:
            pe = float(pair_energy[0].item())
            ct = float(contradiction[0].item())
            lines.append(f"interference  compat={pe:.3f}  conflict={ct:.3f}")
    return "\n".join(lines)


def format_readout_channels_section(inspection: ReadoutChannelInspection) -> str:
    from lmf.core.decode.readout_channels import format_readout_channels_report

    return format_readout_channels_report(inspection)


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

    if inspection.readout_inspection is not None:
        sections.append(format_readout_channels_section(inspection.readout_inspection))

    return "\n".join(sections)


def tensors_to_ranked_lists(
    trace_ids: Tensor,
    scores: Tensor,
    trace_amp: Tensor,
    cue_drive: Tensor,
    *,
    source_cue_id: Tensor | None = None,
    source_token_id: Tensor | None = None,
    source_span: Tensor | None = None,
    cue_type: Tensor | None = None,
    normalized_cue_ids: Tensor | None = None,
) -> tuple[
    list[int],
    list[float],
    list[float],
    list[float],
    list[int] | None,
    list[int] | None,
    list[tuple[int, int]] | None,
    list[str] | None,
    list[int] | None,
]:
    """Sort active traces by match score descending for readable output."""

    ids = trace_ids[0].detach().cpu().tolist()
    score_values = scores[0].detach().cpu().tolist()
    amp_values = trace_amp[0].detach().cpu().tolist()
    drive_values = cue_drive[0].detach().cpu().tolist()

    provenance_fields: list[list[object]] = []
    if (
        source_cue_id is not None
        and source_token_id is not None
        and source_span is not None
        and cue_type is not None
        and normalized_cue_ids is not None
    ):
        provenance_fields = [
            source_cue_id[0].detach().cpu().tolist(),
            source_token_id[0].detach().cpu().tolist(),
            source_span[0].detach().cpu().tolist(),
            [format_cue_type_label(int(value)) for value in cue_type[0].detach().cpu().tolist()],
            normalized_cue_ids[0].detach().cpu().tolist(),
        ]

    ranked = sorted(
        zip(ids, score_values, amp_values, drive_values, *provenance_fields),
        key=lambda row: row[1],
        reverse=True,
    )
    if not ranked:
        return [], [], [], [], None, None, None, None, None

    if provenance_fields:
        (
            ids_out,
            scores_out,
            amps_out,
            drives_out,
            src_cues_out,
            token_ids_out,
            spans_raw,
            cue_types_out,
            norm_ids_out,
        ) = zip(*ranked)
        spans_out = [(int(span[0]), int(span[1])) for span in spans_raw]
        return (
            list(ids_out),
            list(scores_out),
            list(amps_out),
            list(drives_out),
            list(src_cues_out),
            list(token_ids_out),
            spans_out,
            list(cue_types_out),
            list(norm_ids_out),
        )

    ids_out, scores_out, amps_out, drives_out = zip(*ranked)
    return list(ids_out), list(scores_out), list(amps_out), list(drives_out), None, None, None, None, None


__all__ = [
    "PipelineInspection",
    "format_pipeline_report",
    "format_readout_channels_section",
    "format_routing_section",
    "resolve_source_tokens",
    "tensors_to_ranked_lists",
]
