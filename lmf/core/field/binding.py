"""Soft binding layer: sparse relation constraints over active traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from lmf.core.field.types import ContextPressure
from lmf.core.state.types import ActiveRegion, BasinState, BindingState


@dataclass
class BindingLayerConfig:
    content_dim: int
    relation_channels: int = 4
    max_edges_per_trace: int = 8

    def __post_init__(self) -> None:
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.relation_channels <= 0:
            raise ValueError("relation_channels must be positive.")
        if self.max_edges_per_trace <= 0:
            raise ValueError("max_edges_per_trace must be positive.")


class BindingLayer(nn.Module):
    """Score sparse trace-trace relation constraints (no value mixing)."""

    def __init__(self, config: BindingLayerConfig) -> None:
        super().__init__()
        self.config = config
        self.relation_weights = nn.Parameter(
            torch.empty(config.relation_channels, config.content_dim)
        )
        self.relation_bias = nn.Parameter(torch.zeros(config.relation_channels))
        nn.init.normal_(self.relation_weights, mean=0.0, std=0.02)

    def forward(
        self,
        active_region: ActiveRegion,
        basin_state: BasinState,
        context: ContextPressure,
    ) -> BindingState:
        _ = basin_state, context
        content = active_region.trace_content
        if content.dim() != 3:
            raise ValueError("active_region.trace_content must have shape [batch, traces, dim].")

        batch_size, num_traces, dim = content.shape
        if dim != self.config.content_dim:
            raise ValueError("trace content width must match binding content_dim.")

        src = content.unsqueeze(2).expand(batch_size, num_traces, num_traces, dim)
        dst = content.unsqueeze(1).expand(batch_size, num_traces, num_traces, dim)
        pair_features = src * dst

        channel_scores = torch.einsum(
            "btkd,rd->btkr",
            pair_features,
            self.relation_weights,
        ) + self.relation_bias
        pair_strength = channel_scores.max(dim=-1).values

        eye = torch.eye(num_traces, device=content.device, dtype=torch.bool)
        pair_strength = pair_strength.masked_fill(eye.unsqueeze(0), float("-inf"))

        k_keep = min(self.config.max_edges_per_trace, max(num_traces - 1, 1))
        top_values, top_indices = torch.topk(pair_strength, k=k_keep, dim=-1)

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
