"""Context-sensitive binding pair scorer (Commit C2).

Scores whether two active traces should link using trace content, sentence
context, span distance/order, and cue types — not string pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lmf.core.dmf.cue_provenance import POOLED_SOURCE_CUE_ID
from lmf.core.state.types import ActiveRegion


@dataclass
class BindingPairScorerConfig:
    content_dim: int
    context_dim: int
    num_relations: int = 4
    cue_type_count: int = 7
    relation_embed_dim: int = 8
    cue_type_embed_dim: int = 4
    hidden_dim: int | None = None

    def __post_init__(self) -> None:
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if self.num_relations <= 0:
            raise ValueError("num_relations must be positive.")
        if self.cue_type_count <= 0:
            raise ValueError("cue_type_count must be positive.")
        if self.relation_embed_dim <= 0:
            raise ValueError("relation_embed_dim must be positive.")
        if self.cue_type_embed_dim <= 0:
            raise ValueError("cue_type_embed_dim must be positive.")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")

    @property
    def resolved_hidden_dim(self) -> int:
        return self.hidden_dim if self.hidden_dim is not None else max(self.content_dim * 2, 32)


class BindingPairScorer(nn.Module):
    """MLP pair scorer: sigmoid strength per relation channel (no value mixing)."""

    def __init__(self, config: BindingPairScorerConfig) -> None:
        super().__init__()
        self.config = config
        self.relation_embeddings = nn.Embedding(config.num_relations, config.relation_embed_dim)
        self.cue_type_embeddings = nn.Embedding(config.cue_type_count, config.cue_type_embed_dim)

        feature_dim = (
            config.content_dim * 4
            + config.relation_embed_dim
            + config.context_dim
            + 2
            + config.cue_type_embed_dim * 2
        )
        hidden = config.resolved_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.relation_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cue_type_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        active_region: ActiveRegion,
        *,
        context_summary: Tensor,
    ) -> Tensor:
        """Return binding strengths with shape ``[batch, traces, traces, relations]``."""

        content = active_region.trace_content
        if content.dim() != 3:
            raise ValueError("active_region.trace_content must have shape [batch, traces, dim].")
        if content.shape[-1] != self.config.content_dim:
            raise ValueError("trace content width must match content_dim.")

        batch_size, num_traces, _dim = content.shape
        if context_summary.dim() == 1:
            context_summary = context_summary.unsqueeze(0)
        if context_summary.shape != (batch_size, self.config.context_dim):
            raise ValueError("context_summary must have shape [batch, context_dim].")

        span_distance, span_order = self._pair_span_features(active_region, num_traces=num_traces)
        cue_type_i, cue_type_j = self._pair_cue_types(active_region, num_traces=num_traces)

        trace_i = content.unsqueeze(2).expand(batch_size, num_traces, num_traces, _dim)
        trace_j = content.unsqueeze(1).expand(batch_size, num_traces, num_traces, _dim)
        hadamard = trace_i * trace_j
        abs_diff = (trace_i - trace_j).abs()

        context = context_summary.view(batch_size, 1, 1, -1).expand(batch_size, num_traces, num_traces, -1)

        relation_ids = torch.arange(self.config.num_relations, device=content.device)
        relation_embed = self.relation_embeddings(relation_ids)
        relation = relation_embed.view(1, 1, 1, self.config.num_relations, -1)
        relation = relation.expand(batch_size, num_traces, num_traces, self.config.num_relations, -1)

        type_i = self.cue_type_embeddings(cue_type_i.clamp_min(0))
        type_j = self.cue_type_embeddings(cue_type_j.clamp_min(0))
        type_i = type_i.unsqueeze(3).expand(-1, -1, -1, self.config.num_relations, -1)
        type_j = type_j.unsqueeze(3).expand(-1, -1, -1, self.config.num_relations, -1)

        dist = span_distance.unsqueeze(-1).expand(-1, -1, -1, self.config.num_relations)
        order = span_order.unsqueeze(-1).expand(-1, -1, -1, self.config.num_relations)

        base = torch.cat(
            [
                trace_i.unsqueeze(3).expand(-1, -1, -1, self.config.num_relations, -1),
                trace_j.unsqueeze(3).expand(-1, -1, -1, self.config.num_relations, -1),
                hadamard.unsqueeze(3).expand(-1, -1, -1, self.config.num_relations, -1),
                abs_diff.unsqueeze(3).expand(-1, -1, -1, self.config.num_relations, -1),
                relation,
                context.unsqueeze(3).expand(-1, -1, -1, self.config.num_relations, -1),
                dist.unsqueeze(-1),
                order.unsqueeze(-1),
                type_i,
                type_j,
            ],
            dim=-1,
        )

        strengths = torch.sigmoid(self.mlp(base).squeeze(-1))

        eye = torch.eye(num_traces, device=content.device, dtype=torch.bool)
        strengths = strengths.masked_fill(eye.view(1, num_traces, num_traces, 1), 0.0)

        if active_region.mask is not None:
            node_mask = active_region.mask.bool()
            pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
            strengths = strengths * pair_mask.unsqueeze(-1).to(dtype=strengths.dtype)

        return strengths

    def pairwise_mass(
        self,
        active_region: ActiveRegion,
        *,
        context_summary: Tensor,
    ) -> Tensor:
        """Max relation strength per pair: ``[batch, traces, traces]``."""

        strengths = self.forward(active_region, context_summary=context_summary)
        return strengths.max(dim=-1).values

    def _pair_span_features(self, active_region: ActiveRegion, *, num_traces: int) -> tuple[Tensor, Tensor]:
        device = active_region.trace_content.device
        dtype = active_region.trace_content.dtype

        if active_region.source_cue_id is None or active_region.source_span is None:
            zeros = torch.zeros(
                active_region.trace_content.shape[0],
                num_traces,
                num_traces,
                device=device,
                dtype=dtype,
            )
            return zeros, zeros

        source_cue = active_region.source_cue_id.long()
        span = active_region.source_span.to(dtype=dtype)
        span_center = (span[..., 0] + span[..., 1]) * 0.5

        center_i = span_center.unsqueeze(2).expand(-1, num_traces, num_traces)
        center_j = span_center.unsqueeze(1).expand(-1, num_traces, num_traces)
        distance = (center_i - center_j).abs()

        seq_scale = distance.amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
        norm_distance = distance / seq_scale

        cue_i = source_cue.unsqueeze(2).expand(-1, num_traces, num_traces)
        cue_j = source_cue.unsqueeze(1).expand(-1, num_traces, num_traces)
        order = torch.sign(cue_j - cue_i)
        order = torch.where(
            (cue_i == POOLED_SOURCE_CUE_ID) | (cue_j == POOLED_SOURCE_CUE_ID),
            torch.zeros_like(order),
            order,
        )
        return norm_distance, order

    def _pair_cue_types(self, active_region: ActiveRegion, *, num_traces: int) -> tuple[Tensor, Tensor]:
        device = active_region.trace_content.device
        if active_region.cue_type is None:
            zeros = torch.zeros(
                active_region.trace_content.shape[0],
                num_traces,
                num_traces,
                device=device,
                dtype=torch.long,
            )
            return zeros, zeros

        cue_type = active_region.cue_type.long()
        type_i = cue_type.unsqueeze(2).expand(-1, num_traces, num_traces)
        type_j = cue_type.unsqueeze(1).expand(-1, num_traces, num_traces)
        return type_i, type_j


__all__ = ["BindingPairScorer", "BindingPairScorerConfig"]
