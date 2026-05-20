"""Convert binding constraints into trace and basin forces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lmf.core.basin.basin_forces import BasinForceComposer, BasinForceComposerConfig
from lmf.core.basin.binding_edges import validate_binding_state
from lmf.core.field.types import BindingForces, ContextPressure
from lmf.core.state.types import ActiveRegion, BasinState, BindingState


@dataclass
class BindingForcesConfig:
    num_basins: int
    content_dim: int
    num_relations: int = 4
    force_scale: float = 1.0
    direct_trace_to_basin_scale: float = 0.2
    bound_pair_to_basin_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.num_relations <= 0:
            raise ValueError("num_relations must be positive.")
        if self.force_scale <= 0.0:
            raise ValueError("force_scale must be positive.")
        if self.direct_trace_to_basin_scale < 0.0:
            raise ValueError("direct_trace_to_basin_scale must be non-negative.")
        if self.bound_pair_to_basin_scale < 0.0:
            raise ValueError("bound_pair_to_basin_scale must be non-negative.")


class BindingForcesModule(nn.Module):
    """Turn binding edges into forces. Does not mix trace content vectors."""

    def __init__(self, config: BindingForcesConfig) -> None:
        super().__init__()
        self.config = config
        self.src_scale = nn.Parameter(torch.tensor(1.0))
        self.dst_scale = nn.Parameter(torch.tensor(1.0))
        self.basin_composer = BasinForceComposer(
            BasinForceComposerConfig(
                num_basins=config.num_basins,
                content_dim=config.content_dim,
                num_relations=config.num_relations,
                direct_trace_to_basin_scale=config.direct_trace_to_basin_scale,
                bound_pair_to_basin_scale=config.bound_pair_to_basin_scale,
            )
        )

    @property
    def pair_basin_support(self):
        """Backward-compatible access to the pair-derived basin module."""

        return self.basin_composer.pair_basin_support

    @property
    def basin_projection(self) -> Tensor:
        """Backward-compatible access to the direct-path basin weights."""

        return self.basin_composer.direct_projection

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

        validate_binding_state(
            binding_state,
            batch_size=batch_size,
            num_traces=num_traces,
            num_relations=self.config.num_relations,
            require_relation_index=False,
        )

        src = edge_index[:, 0, :]
        dst = edge_index[:, 1, :]
        weighted = strengths * self.config.force_scale

        trace_force.scatter_add_(1, src, weighted * self.src_scale)
        trace_force.scatter_add_(1, dst, weighted * self.dst_scale)

        basin_breakdown = self.basin_composer(active_region, basin_state, binding_state)

        stability_force = -trace_force.abs().mean(dim=-1)

        return BindingForces(
            trace_force=trace_force,
            basin_force=basin_breakdown.total,
            stability_force=stability_force,
            direct_basin_force=basin_breakdown.direct,
            bound_pair_basin_force=basin_breakdown.bound_pair,
        )


# Commit plan name: field_loop.binding_forces
BindingForcesLayer = BindingForcesModule

__all__ = ["BindingForcesConfig", "BindingForcesLayer", "BindingForcesModule"]
