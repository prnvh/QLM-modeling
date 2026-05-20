"""Build normalized ``CognitiveState`` packets from field-loop outputs (Commit E)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from torch import Tensor

from lmf.core.field.types import FieldLoopOutput
from lmf.core.state.types import (
    ActiveRegion,
    BasinState,
    BindingState,
    CognitiveState,
    CuePacket,
    InterferenceState,
    LucidityState,
)


def build_cognitive_state(
    *,
    cue_packet: CuePacket,
    active_region: ActiveRegion,
    basin_state: BasinState,
    binding_state: BindingState | None = None,
    interference_state: InterferenceState | None = None,
    lucidity_state: LucidityState | None = None,
    context_summary: Tensor | None = None,
    meta: Mapping[str, Any] | None = None,
) -> CognitiveState:
    """Assemble a ``CognitiveState`` for decoder / lucidity consumers.

    ``context_summary`` must come from ``ContextOp`` (pooled prompt pressure), not
    raw token embeddings. It is stored on ``meta['context_summary']`` for the
    decoder context channel.
    """

    merged_meta: dict[str, Any] = dict(meta or {})
    if context_summary is not None:
        if context_summary.dim() == 1:
            context_summary = context_summary.unsqueeze(0)
        merged_meta["context_summary"] = context_summary

    return CognitiveState(
        cue_packet=cue_packet,
        active_region=active_region,
        basin_state=basin_state,
        binding_state=binding_state,
        interference_state=interference_state,
        lucidity_state=lucidity_state,
        meta=merged_meta,
    )


def build_cognitive_state_from_field_loop(
    *,
    cue_packet: CuePacket,
    active_region: ActiveRegion,
    basin_state: BasinState,
    field_output: FieldLoopOutput,
    context_summary: Tensor | None = None,
    lucidity_state: LucidityState | None = None,
    meta: Mapping[str, Any] | None = None,
) -> CognitiveState:
    """Merge settled field-loop outputs into a ``CognitiveState``."""

    settled_region = replace(active_region, trace_amp=field_output.active_region_trace_amp)
    settled_basin = replace(basin_state, pressures=field_output.basin_pressures)
    binding_state = field_output.binding_state
    if not isinstance(binding_state, BindingState):
        raise TypeError("field_output.binding_state must be a BindingState.")
    interference_state = field_output.interference_state
    if not isinstance(interference_state, InterferenceState):
        raise TypeError("field_output.interference_state must be an InterferenceState.")

    merged_meta: dict[str, Any] = dict(meta or {})
    merged_meta.setdefault("settling_steps", field_output.steps_run)

    return build_cognitive_state(
        cue_packet=cue_packet,
        active_region=settled_region,
        basin_state=settled_basin,
        binding_state=binding_state,
        interference_state=interference_state,
        lucidity_state=lucidity_state,
        context_summary=context_summary,
        meta=merged_meta,
    )


def require_context_summary(state: CognitiveState) -> Tensor:
    """Return ``context_summary`` stored on the cognitive state."""

    summary = state.meta.get("context_summary")
    if summary is None:
        raise ValueError(
            "CognitiveState.meta must include 'context_summary' from ContextOp "
            "for the decoder context channel."
        )
    if not isinstance(summary, Tensor):
        raise TypeError("context_summary must be a torch.Tensor.")
    if summary.dim() == 1:
        return summary.unsqueeze(0)
    return summary


__all__ = [
    "build_cognitive_state",
    "build_cognitive_state_from_field_loop",
    "require_context_summary",
]
