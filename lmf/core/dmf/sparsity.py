"""Sparsity helpers for top-k trace routing and inspection metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def resolve_top_k(*, requested_k: int, num_traces: int) -> int:
    """Clamp top-k to the valid range for a trace bank."""

    if requested_k <= 0:
        raise ValueError("top_k must be positive.")
    if num_traces <= 0:
        raise ValueError("num_traces must be positive.")
    return min(requested_k, num_traces)


def topk_trace_routing(
    scores: Tensor,
    *,
    top_k: int,
) -> tuple[Tensor, Tensor]:
    """Select the highest-scoring trace ids for each batch row.

    Args:
        scores: Tensor shaped ``[batch, num_traces]``.
        top_k: Number of traces to keep per batch row.

    Returns:
        ``(trace_ids, top_scores)`` with shape ``[batch, top_k]``.
    """

    if scores.dim() != 2:
        raise ValueError("scores must have shape [batch, num_traces].")
    if scores.numel() == 0:
        raise ValueError("scores must not be empty.")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite.")

    batch_size, num_traces = scores.shape
    k = resolve_top_k(requested_k=top_k, num_traces=num_traces)
    top_scores, trace_ids = torch.topk(scores, k=k, dim=-1)
    if trace_ids.shape != (batch_size, k):
        raise RuntimeError("top-k routing returned an unexpected shape.")
    return trace_ids.long(), top_scores


def active_trace_fraction(*, active_traces: int, num_traces: int) -> float:
    """Return the fraction of the bank that is active for one routing step."""

    if active_traces < 0:
        raise ValueError("active_traces must be non-negative.")
    if num_traces <= 0:
        raise ValueError("num_traces must be positive.")
    return active_traces / num_traces


@dataclass(frozen=True)
class SparsityReport:
    """Human-readable sparsity summary for one routing step."""

    batch_size: int
    num_traces: int
    active_traces: int
    active_fraction: float
    inactive_traces: int

    @classmethod
    def from_routing(cls, *, batch_size: int, num_traces: int, active_traces: int) -> SparsityReport:
        k = resolve_top_k(requested_k=active_traces, num_traces=num_traces)
        fraction = active_trace_fraction(active_traces=k, num_traces=num_traces)
        return cls(
            batch_size=batch_size,
            num_traces=num_traces,
            active_traces=k,
            active_fraction=fraction,
            inactive_traces=num_traces - k,
        )


__all__ = [
    "SparsityReport",
    "active_trace_fraction",
    "resolve_top_k",
    "topk_trace_routing",
]
