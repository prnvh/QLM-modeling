"""Soft binding layer: sparse relation constraints over active traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lmf.core.field.binding_pair_scorer import BindingPairScorer, BindingPairScorerConfig
from lmf.core.field.types import ContextPressure
from lmf.core.state.types import ActiveRegion, BasinState, BindingState


@dataclass
class BindingLayerConfig:
    content_dim: int
    context_dim: int
    relation_channels: int = 4
    max_edges_per_trace: int = 8

    def __post_init__(self) -> None:
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if self.relation_channels <= 0:
            raise ValueError("relation_channels must be positive.")
        if self.max_edges_per_trace <= 0:
            raise ValueError("max_edges_per_trace must be positive.")


class BindingLayer(nn.Module):
    """Score sparse trace-trace relation constraints (no value mixing)."""

    def __init__(self, config: BindingLayerConfig) -> None:
        super().__init__()
        self.config = config
        self.pair_scorer = BindingPairScorer(
            BindingPairScorerConfig(
                content_dim=config.content_dim,
                context_dim=config.context_dim,
                num_relations=config.relation_channels,
            )
        )

    def forward(
        self,
        active_region: ActiveRegion,
        basin_state: BasinState,
        context: ContextPressure,
    ) -> BindingState:
        _ = basin_state
        pair_strength = self.pair_scorer.forward(
            active_region,
            context_summary=context.context_summary,
        )
        return self._sparse_binding_state(active_region, pair_strength)

    def pairwise_strength_matrix(
        self,
        active_region: ActiveRegion,
        *,
        context: ContextPressure,
    ) -> Tensor:
        """Return max-relation binding mass ``[batch, traces, traces]``."""

        return self.pair_scorer.pairwise_mass(
            active_region,
            context_summary=context.context_summary,
        )

    def _sparse_binding_state(
        self,
        active_region: ActiveRegion,
        pair_strength: Tensor,
    ) -> BindingState:
        content = active_region.trace_content
        batch_size, num_traces, _dim = content.shape

        pair_mass = pair_strength.max(dim=-1).values
        eye = torch.eye(num_traces, device=content.device, dtype=torch.bool)
        pair_mass = pair_mass.masked_fill(eye.unsqueeze(0), float("-inf"))

        k_keep = min(self.config.max_edges_per_trace, max(num_traces - 1, 1))
        top_values, top_indices = torch.topk(pair_mass, k=k_keep, dim=-1)

        src_ids = torch.arange(num_traces, device=content.device).view(1, num_traces, 1)
        src_ids = src_ids.expand(batch_size, num_traces, k_keep).reshape(batch_size, -1)
        dst_ids = top_indices.reshape(batch_size, -1)
        strengths = top_values.reshape(batch_size, -1)

        valid = torch.isfinite(strengths)
        edge_index = torch.stack([src_ids, dst_ids], dim=1)
        relation_strength = strengths

        centrality = strengths.clamp_min(0.0).sum(dim=-1)
        centrality = centrality / centrality.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if active_region.mask is not None:
            node_mask = active_region.mask
            edge_mask = node_mask.gather(1, src_ids) & node_mask.gather(1, dst_ids)
            valid = valid & edge_mask
            relation_strength = relation_strength.masked_fill(~valid, 0.0)

        return BindingState(
            edge_index=edge_index,
            relation_strength=relation_strength,
            centrality=centrality,
        )


__all__ = ["BindingLayer", "BindingLayerConfig"]
