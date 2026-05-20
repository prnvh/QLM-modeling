"""Binding-edge validation and trace-content gathering for basin routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lmf.core.state.types import ActiveRegion, BindingState


@dataclass(frozen=True)
class BindingEdgeBatch:
    """Validated sparse edges with gathered trace content."""

    src_content: Tensor
    dst_content: Tensor
    relation_strength: Tensor
    relation_index: Tensor
    num_edges: int


def validate_binding_state(
    binding_state: BindingState,
    *,
    batch_size: int,
    num_traces: int,
    num_relations: int,
    require_relation_index: bool = True,
) -> int:
    """Validate sparse binding edges; return edge count."""

    edge_index = binding_state.edge_index
    strengths = binding_state.relation_strength

    if edge_index.dim() != 3 or edge_index.shape[1] != 2:
        raise ValueError("binding_state.edge_index must have shape [batch, 2, num_edges].")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("binding_state.edge_index must be integer typed.")
    if edge_index.shape[0] != batch_size:
        raise ValueError("binding_state batch size must match active_region batch size.")

    num_edges = int(edge_index.shape[2])
    if strengths.shape != (batch_size, num_edges):
        raise ValueError("binding_state.relation_strength must have shape [batch, num_edges].")
    if not strengths.dtype.is_floating_point:
        raise ValueError("binding_state.relation_strength must be a floating tensor.")

    if num_edges == 0:
        if require_relation_index and binding_state.relation_index is not None:
            if binding_state.relation_index.shape != (batch_size, 0):
                raise ValueError("binding_state.relation_index must be empty when there are no edges.")
        return 0

    src = edge_index[:, 0, :]
    dst = edge_index[:, 1, :]
    if (src < 0).any() or (dst < 0).any():
        raise ValueError("binding edge indices must be non-negative.")
    if (src >= num_traces).any() or (dst >= num_traces).any():
        raise ValueError("binding edge indices must be less than num_traces.")

    if require_relation_index and binding_state.relation_index is None:
        raise ValueError("binding_state.relation_index is required for pair-derived basin support.")

    if binding_state.relation_index is not None:
        relation_index = binding_state.relation_index
        if relation_index.shape != (batch_size, num_edges):
            raise ValueError("binding_state.relation_index must have shape [batch, num_edges].")
        if relation_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("binding_state.relation_index must be integer typed.")
        if (relation_index < 0).any() or (relation_index >= num_relations).any():
            raise ValueError("binding_state.relation_index must be in [0, num_relations).")

    if not torch.isfinite(strengths).all():
        raise ValueError("binding_state.relation_strength contains non-finite values.")

    return num_edges


def gather_binding_edge_batch(
    active_region: ActiveRegion,
    binding_state: BindingState,
    *,
    content_dim: int,
    num_relations: int,
) -> BindingEdgeBatch:
    """Gather per-edge trace content after validating binding topology."""

    batch_size, num_traces, region_dim = active_region.trace_content.shape
    if region_dim != content_dim:
        raise ValueError("active_region.trace_content width must match content_dim.")

    num_edges = validate_binding_state(
        binding_state,
        batch_size=batch_size,
        num_traces=num_traces,
        num_relations=num_relations,
        require_relation_index=True,
    )

    if num_edges == 0:
        device = active_region.trace_content.device
        dtype = active_region.trace_content.dtype
        return BindingEdgeBatch(
            src_content=torch.zeros(batch_size, 0, content_dim, device=device, dtype=dtype),
            dst_content=torch.zeros(batch_size, 0, content_dim, device=device, dtype=dtype),
            relation_strength=torch.zeros(batch_size, 0, device=device, dtype=dtype),
            relation_index=torch.zeros(batch_size, 0, device=device, dtype=torch.long),
            num_edges=0,
        )

    edge_index = binding_state.edge_index
    src = edge_index[:, 0, :]
    dst = edge_index[:, 1, :]
    gather_index = src.unsqueeze(-1).expand(-1, -1, content_dim)
    src_content = torch.gather(active_region.trace_content, 1, gather_index)
    gather_index = dst.unsqueeze(-1).expand(-1, -1, content_dim)
    dst_content = torch.gather(active_region.trace_content, 1, gather_index)

    relation_index = binding_state.relation_index
    if relation_index is None:
        raise ValueError("binding_state.relation_index is required.")

    return BindingEdgeBatch(
        src_content=src_content,
        dst_content=dst_content,
        relation_strength=binding_state.relation_strength,
        relation_index=relation_index.long(),
        num_edges=num_edges,
    )


__all__ = [
    "BindingEdgeBatch",
    "gather_binding_edge_batch",
    "validate_binding_state",
]
