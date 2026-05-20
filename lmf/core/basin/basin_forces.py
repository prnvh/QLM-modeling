"""Basin force composition: direct trace path + bound-pair path (D1/D2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lmf.core.basin.basin_state import validate_basin_state
from lmf.core.basin.binding_edges import gather_binding_edge_batch
from lmf.core.basin.pair_basin_support import PairDerivedBasinSupport, PairDerivedBasinSupportConfig
from lmf.core.state.types import ActiveRegion, BasinState, BindingState


@dataclass
class BasinForceComposerConfig:
    num_basins: int
    content_dim: int
    num_relations: int
    direct_trace_to_basin_scale: float = 0.2
    bound_pair_to_basin_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.num_relations <= 0:
            raise ValueError("num_relations must be positive.")
        if self.direct_trace_to_basin_scale < 0.0:
            raise ValueError("direct_trace_to_basin_scale must be non-negative.")
        if self.bound_pair_to_basin_scale < 0.0:
            raise ValueError("bound_pair_to_basin_scale must be non-negative.")


@dataclass
class BasinForceBreakdown:
    """Decomposed basin forces for logging, tests, and ablations."""

    total: Tensor
    direct: Tensor
    bound_pair: Tensor


class BasinForceComposer(nn.Module):
    """Combine weak direct-trace and primary bound-pair basin pressure paths."""

    def __init__(self, config: BasinForceComposerConfig) -> None:
        super().__init__()
        self.config = config
        self.direct_projection = nn.Parameter(torch.empty(config.num_basins, 1))
        self.pair_basin_support = PairDerivedBasinSupport(
            PairDerivedBasinSupportConfig(
                content_dim=config.content_dim,
                num_basins=config.num_basins,
                num_relations=config.num_relations,
            )
        )
        nn.init.normal_(self.direct_projection, mean=0.0, std=0.02)

    def forward(
        self,
        active_region: ActiveRegion,
        basin_state: BasinState,
        binding_state: BindingState,
    ) -> BasinForceBreakdown:
        validate_basin_state(basin_state)

        direct = self._direct_trace_basin_force(active_region, basin_state)
        bound_pair = self._bound_pair_basin_force(active_region, basin_state, binding_state)
        total = (
            self.config.direct_trace_to_basin_scale * direct
            + self.config.bound_pair_to_basin_scale * bound_pair
        )

        if total.shape != basin_state.pressures.shape:
            raise ValueError("composed basin force must match basin_state.pressures shape.")
        if not torch.isfinite(total).all():
            raise ValueError("composed basin force contains non-finite values.")

        return BasinForceBreakdown(total=total, direct=direct, bound_pair=bound_pair)

    def _direct_trace_basin_force(self, active_region: ActiveRegion, basin_state: BasinState) -> Tensor:
        trace_amp = active_region.trace_amp
        if trace_amp.dim() != 2:
            raise ValueError("active_region.trace_amp must have shape [batch, num_traces].")
        if trace_amp.shape[1] != active_region.trace_content.shape[1]:
            raise ValueError("trace_amp trace count must match trace_content.")

        if active_region.mask is not None:
            mask = active_region.mask.to(dtype=trace_amp.dtype)
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pooled_trace = (trace_amp * mask).sum(dim=-1, keepdim=True) / denom
        else:
            pooled_trace = trace_amp.mean(dim=-1, keepdim=True)

        basin_weights = self.direct_projection.squeeze(-1).unsqueeze(0)
        direct = pooled_trace * basin_weights
        if direct.shape != basin_state.pressures.shape:
            direct = direct.expand_as(basin_state.pressures)
        return direct

    def _bound_pair_basin_force(
        self,
        active_region: ActiveRegion,
        basin_state: BasinState,
        binding_state: BindingState,
    ) -> Tensor:
        edges = gather_binding_edge_batch(
            active_region,
            binding_state,
            content_dim=self.config.content_dim,
            num_relations=self.config.num_relations,
        )
        if edges.num_edges == 0:
            return torch.zeros_like(basin_state.pressures)

        bound_pair = self.pair_basin_support(
            edges.src_content,
            edges.dst_content,
            relation_index=edges.relation_index,
            binding_strength=edges.relation_strength,
        )
        if bound_pair.shape != basin_state.pressures.shape:
            raise ValueError("bound_pair basin force must match basin_state.pressures shape.")
        return bound_pair


__all__ = [
    "BasinForceBreakdown",
    "BasinForceComposer",
    "BasinForceComposerConfig",
]
