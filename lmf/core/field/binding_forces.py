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
    content_dim: int
    force_scale: float = 1.0
    direct_trace_to_basin_scale: float = 0.2
    bound_pair_to_basin_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
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
        self.basin_projection = nn.Parameter(torch.empty(config.num_basins, 1))
        pair_input_dim = config.content_dim * 3
        self.pair_basin_projection = nn.Linear(pair_input_dim, config.num_basins, bias=True)
        nn.init.normal_(self.basin_projection, mean=0.0, std=0.02)
        nn.init.normal_(self.pair_basin_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.pair_basin_projection.bias)

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

        direct_basin = self._direct_trace_basin_force(active_region, basin_state)
        bound_pair_basin = self._bound_pair_basin_force(active_region, binding_state, basin_state)
        basin_force = (
            self.config.direct_trace_to_basin_scale * direct_basin
            + self.config.bound_pair_to_basin_scale * bound_pair_basin
        )

        stability_force = -trace_force.abs().mean(dim=-1)

        return BindingForces(
            trace_force=trace_force,
            basin_force=basin_force,
            stability_force=stability_force,
            direct_basin_force=direct_basin,
            bound_pair_basin_force=bound_pair_basin,
        )

    def _direct_trace_basin_force(self, active_region: ActiveRegion, basin_state: BasinState) -> Tensor:
        """Weak path: isolated active trace amplitudes pooled into basin pressure."""

        pooled_trace = active_region.trace_amp.mean(dim=-1, keepdim=True)
        basin_weights = self.basin_projection.squeeze(-1).unsqueeze(0)
        direct = pooled_trace * basin_weights
        if direct.shape != basin_state.pressures.shape:
            direct = direct.expand_as(basin_state.pressures)
        return direct

    def _bound_pair_basin_force(
        self,
        active_region: ActiveRegion,
        binding_state: BindingState,
        basin_state: BasinState,
    ) -> Tensor:
        """Primary path: bound trace pairs aggregate into basin pressure."""

        content = active_region.trace_content
        batch_size, _num_traces, content_dim = content.shape
        if content_dim != self.config.content_dim:
            raise ValueError("active_region.trace_content width must match content_dim.")

        edge_index = binding_state.edge_index
        strengths = binding_state.relation_strength
        num_edges = edge_index.shape[-1]

        src = edge_index[:, 0, :]
        dst = edge_index[:, 1, :]
        gather_index = src.unsqueeze(-1).expand(-1, -1, content_dim)
        src_content = torch.gather(content, 1, gather_index)
        gather_index = dst.unsqueeze(-1).expand(-1, -1, content_dim)
        dst_content = torch.gather(content, 1, gather_index)

        pair_features = torch.cat([src_content, dst_content, src_content * dst_content], dim=-1)
        edge_basin = self.pair_basin_projection(pair_features)
        edge_weights = strengths.clamp_min(0.0).unsqueeze(-1)
        bound_pair = (edge_basin * edge_weights).sum(dim=1)

        if bound_pair.shape != basin_state.pressures.shape:
            raise ValueError("bound_pair_basin_force must match basin_state.pressures shape.")

        if num_edges == 0:
            bound_pair = torch.zeros_like(basin_state.pressures)

        return bound_pair


# Commit plan name: field_loop.binding_forces
BindingForcesLayer = BindingForcesModule

__all__ = ["BindingForcesConfig", "BindingForcesLayer", "BindingForcesModule"]
