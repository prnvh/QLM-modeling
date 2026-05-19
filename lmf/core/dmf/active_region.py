"""Build sparse ActiveRegion packets from routed traces."""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from lmf.core.dmf.cue_provenance import preview_provenance_row
from lmf.core.dmf.trace_bank import TraceBank
from lmf.core.dmf.trace_router import RoutingResult
from lmf.core.input.cue_packet import CuePacket
from lmf.core.state.types import ActiveRegion

LOGGER = logging.getLogger(__name__)


def build_active_region(
    *,
    trace_bank: TraceBank,
    routing: RoutingResult,
    cue_packet: CuePacket,
) -> ActiveRegion:
    """Materialize the blue active trace region from a routing result.

    The full trace bank is never copied. Only the selected trace rows are
    gathered into the returned packet. Source metadata from routing is copied
    onto each active trace for later binding supervision.
    """

    _ = cue_packet
    trace_ids = _normalize_trace_ids(routing.trace_ids, num_traces=trace_bank.config.num_traces)
    batch_size, top_k = trace_ids.shape

    gathered = trace_bank.state(trace_ids=trace_ids.reshape(-1))
    trace_content = gathered.content.reshape(batch_size, top_k, -1)

    trace_amp = torch.sigmoid(routing.scores)
    cue_drive = routing.scores
    mask = torch.isfinite(routing.scores)

    if not bool(mask.all()):
        raise ValueError("routing scores must be finite for all selected traces.")

    _log_active_region(
        trace_ids=trace_ids,
        routing=routing,
        batch_size=batch_size,
        top_k=top_k,
    )

    return ActiveRegion(
        trace_ids=trace_ids,
        trace_content=trace_content,
        trace_amp=trace_amp,
        cue_drive=cue_drive,
        mask=mask,
        source_cue_id=routing.source_cue_id,
        source_token_id=routing.source_token_id,
        source_span=routing.source_span,
        cue_type=routing.cue_type,
        normalized_cue_ids=routing.normalized_cue_ids,
    )


def _normalize_trace_ids(trace_ids: Tensor, *, num_traces: int) -> Tensor:
    if trace_ids.dim() != 2:
        raise ValueError("trace_ids must have shape [batch, top_k].")
    if trace_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("trace_ids must be an integer tensor.")
    if trace_ids.numel() == 0:
        raise ValueError("trace_ids must not be empty.")
    trace_ids = trace_ids.long()
    min_id = int(trace_ids.min().item())
    max_id = int(trace_ids.max().item())
    if min_id < 0 or max_id >= num_traces:
        raise ValueError("trace_ids contain ids outside the configured trace bank.")
    return trace_ids


def _log_active_region(
    *,
    trace_ids: Tensor,
    routing: RoutingResult,
    batch_size: int,
    top_k: int,
) -> None:
    if not LOGGER.isEnabledFor(logging.INFO):
        return

    rows: list[str] = []
    for index in range(min(top_k, 8)):
        span = routing.source_span[0, index].detach().cpu().tolist()
        rows.append(
            preview_provenance_row(
                source_cue_id=int(routing.source_cue_id[0, index].item()),
                source_token_id=int(routing.source_token_id[0, index].item()),
                source_span=(int(span[0]), int(span[1])),
                cue_type=int(routing.cue_type[0, index].item()),
                normalized_cue_id=int(routing.normalized_cue_ids[0, index].item()),
            )
        )

    LOGGER.info(
        "active_region.build batch_size=%s top_k=%s trace_ids=%s provenance=%s",
        batch_size,
        top_k,
        trace_ids.detach().cpu().tolist(),
        " | ".join(rows),
    )


class ActiveRegionBuilder(nn.Module):
    """Thin module wrapper around :func:`build_active_region`."""

    def forward(
        self,
        trace_bank: TraceBank,
        routing: RoutingResult,
        cue_packet: CuePacket,
    ) -> ActiveRegion:
        return build_active_region(
            trace_bank=trace_bank,
            routing=routing,
            cue_packet=cue_packet,
        )


__all__ = [
    "ActiveRegionBuilder",
    "build_active_region",
]
