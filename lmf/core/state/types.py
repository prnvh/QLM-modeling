"""Shared structural types for tensors passed through the LMF stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from torch import Tensor


@dataclass
class CuePacket:
    """Token- or chunk-level cues driving the trace router and field."""

    cues: Tensor
    mask: Tensor | None = None
    positions: Tensor | None = None
    pooled: Tensor | None = None
    token_ids: Tensor | None = None
    special_token_ids: tuple[int, int, int, int, int] | None = None


@dataclass
class ActiveRegion:
    """Sparse DMF subset selected for this step (per plan: trace_ids, contents, amps, drive)."""

    trace_ids: Tensor
    trace_content: Tensor
    trace_amp: Tensor
    cue_drive: Tensor
    mask: Tensor | None = None
    source_cue_id: Tensor | None = None
    source_token_id: Tensor | None = None
    source_span: Tensor | None = None
    cue_type: Tensor | None = None
    normalized_cue_ids: Tensor | None = None


@dataclass
class BindingState:
    """Sparse relational constraints among active traces (not mixed attention outputs)."""

    edge_index: Tensor
    relation_strength: Tensor
    centrality: Tensor | None = None


@dataclass
class InterferenceState:
    """Compatibility / conflict terms feeding the energy, typically binding-gated."""

    pair_energy: Tensor | None = None
    local_energy: Tensor | None = None
    contradiction: Tensor | None = None


@dataclass
class BasinState:
    """Sparse basin / attractor pressures paired with latent basin vectors."""

    pressures: Tensor
    vectors: Tensor
    basin_ids: Tensor | None = None


@dataclass
class LucidityState:
    """Diagnostics for whether the field is settled, ambiguous, or unstable."""

    score: Tensor
    stability: Tensor | None = None
    ambiguity: Tensor | None = None


@dataclass
class CognitiveState:
    """One step of runnable cognitive state threading cue → traces → basins."""

    cue_packet: CuePacket
    active_region: ActiveRegion
    basin_state: BasinState
    binding_state: BindingState | None = None
    interference_state: InterferenceState | None = None
    lucidity_state: LucidityState | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TrainingBatch:
    """Canonical training minibatch tensors (language modeling-style)."""

    input_ids: Tensor
    attention_mask: Tensor | None = None
    target_ids: Tensor | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "CuePacket",
    "ActiveRegion",
    "BindingState",
    "InterferenceState",
    "BasinState",
    "LucidityState",
    "CognitiveState",
    "TrainingBatch",
]
