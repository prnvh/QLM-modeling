"""Field-step datatypes (pressures and forces, not mixed attention outputs)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ContextPressure:
    """Prompt-level drive on the active field."""

    trace_drive: Tensor
    basin_drive: Tensor
    threshold_shift: Tensor
    context_summary: Tensor


@dataclass
class BindingForces:
    """Forces derived from binding constraints (not value mixing)."""

    trace_force: Tensor
    basin_force: Tensor
    stability_force: Tensor
    direct_basin_force: Tensor | None = None
    bound_pair_basin_force: Tensor | None = None


@dataclass
class FieldLoopOutput:
    """State after one or more settling steps."""

    active_region_trace_amp: Tensor
    basin_pressures: Tensor
    binding_state: object
    interference_state: object
    steps_run: int
