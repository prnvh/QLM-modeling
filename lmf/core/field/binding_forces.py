"""Convert binding constraints into trace and basin forces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lmf.core.field.types import BindingForces, ContextPressure
from lmf.core.state.types import ActiveRegion, BasinState, BindingState


@dataclass
class BindingForcesConfig:
    num_basins: int
    force_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.force_scale <= 0.0:
            raise ValueError("force_scale must be positive.")


class BindingForcesModule(nn.Module):
    """Turn binding edges into forces. Does not mix trace content vectors."""

    def __init__(self, config: BindingForcesConfig) -> None:
        super().__init__()
        self.config = config
        self.src_scale = nn.Parameter(torch.tensor(1.0))
        self.dst_scale = nn.Parameter(torch.tensor(1.0))
        self.basin_projection = nn.Parameter(torch.empty(config.num_basins, 1))
        nn.init.normal_(self.basin_projection, mean=0.0, std=0.02)

    def forward(
        self,
        active_region: ActiveRegion,
        basin_state: BasinState,
        binding_state: BindingState,
        context: ContextPressure,
    ) -> BindingForces:
        _ = context
        batch_size, num_traces = active_region.trace_amp.shape
        device = active_region.trace_amp.device
        dtype = active_region.trace_amp.dtype

        trace_force = torch.zeros(batch_size, num_traces, device=device, dtype=dtype)
        edge_index = binding_state.edge_index
        strengths = binding_state.relation_strength

        if edge_index.dim() != 3 or edge_index.shape[1] != 2:
            raise ValueError("binding_state.edge_index must have shape [batch, 2, num_edges].")

        src = edge_index[:, 0, :]
        dst = edge_index[:, 1, :]
        weighted = strengths * self.config.force_scale

        trace_force.scatter_add_(1, src, weighted * self.src_scale)
        trace_force.scatter_add_(1, dst, weighted * self.dst_scale)

        pooled_trace = trace_force.mean(dim=-1, keepdim=True)
        basin_weights = self.basin_projection.squeeze(-1).unsqueeze(0)
        basin_force = pooled_trace * basin_weights
        if basin_force.shape != basin_state.pressures.shape:
            basin_force = basin_force.expand_as(basin_state.pressures)

        stability_force = -trace_force.abs().mean(dim=-1)

        return BindingForces(
            trace_force=trace_force,
            basin_force=basin_force,
            stability_force=stability_force,
        )


# Commit plan name: field_loop.binding_forces
BindingForcesLayer = BindingForcesModule

__all__ = ["BindingForcesConfig", "BindingForcesLayer", "BindingForcesModule"]
