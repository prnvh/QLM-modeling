"""Cue provenance helpers for sparse trace routing (Commit C1).

Each active trace remembers which input cue drove its selection so later
binding supervision can use span distance, cue type, and stable cue ids.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Integer cue-type codes stored on tensors. Labels are for logs and CLI only.
CUE_TYPE_PAD = 0
CUE_TYPE_UNK = 1
CUE_TYPE_MASK = 2
CUE_TYPE_BOS = 3
CUE_TYPE_EOS = 4
CUE_TYPE_TOKEN = 5
CUE_TYPE_POOLED = 6

CUE_TYPE_LABELS: dict[int, str] = {
    CUE_TYPE_PAD: "pad",
    CUE_TYPE_UNK: "unk",
    CUE_TYPE_MASK: "mask",
    CUE_TYPE_BOS: "bos",
    CUE_TYPE_EOS: "eos",
    CUE_TYPE_TOKEN: "token",
    CUE_TYPE_POOLED: "pooled",
}

POOLED_SOURCE_CUE_ID = -1
POOLED_SOURCE_TOKEN_ID = -1
POOLED_NORMALIZED_CUE_ID = -1


@dataclass(frozen=True)
class SpecialTokenIds:
    """Vocabulary ids for special tokens used to classify cue types."""

    pad_id: int
    unk_id: int
    mask_id: int
    bos_id: int
    eos_id: int

    def __post_init__(self) -> None:
        ids = (self.pad_id, self.unk_id, self.mask_id, self.bos_id, self.eos_id)
        if any(value < 0 for value in ids):
            raise ValueError("special token ids must be non-negative.")


@dataclass(frozen=True)
class TraceProvenance:
    """Per-active-trace source metadata carried through routing."""

    source_cue_id: Tensor
    source_token_id: Tensor
    source_span: Tensor
    cue_type: Tensor
    normalized_cue_ids: Tensor

    def __post_init__(self) -> None:
        if self.source_cue_id.shape != self.source_token_id.shape:
            raise ValueError("source_cue_id and source_token_id must share shape.")
        if self.source_cue_id.shape != self.cue_type.shape:
            raise ValueError("source_cue_id and cue_type must share shape.")
        if self.source_cue_id.shape != self.normalized_cue_ids.shape:
            raise ValueError("source_cue_id and normalized_cue_ids must share shape.")
        if self.source_span.shape[:-1] != self.source_cue_id.shape:
            raise ValueError("source_span must have shape [batch, top_k, 2].")
        if self.source_span.shape[-1] != 2:
            raise ValueError("source_span last dimension must be 2.")


def classify_cue_types(token_ids: Tensor, special: SpecialTokenIds) -> Tensor:
    """Map token ids to cue-type codes with the same shape as ``token_ids``."""

    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be an integer tensor.")

    types = torch.full_like(token_ids, CUE_TYPE_TOKEN, dtype=torch.long)
    types = types.masked_fill(token_ids.eq(special.pad_id), CUE_TYPE_PAD)
    types = types.masked_fill(token_ids.eq(special.unk_id), CUE_TYPE_UNK)
    types = types.masked_fill(token_ids.eq(special.mask_id), CUE_TYPE_MASK)
    types = types.masked_fill(token_ids.eq(special.bos_id), CUE_TYPE_BOS)
    types = types.masked_fill(token_ids.eq(special.eos_id), CUE_TYPE_EOS)
    return types


def normalize_cue_ids(token_ids: Tensor, cue_types: Tensor) -> Tensor:
    """Return binding-stable cue ids.

    Content tokens keep their vocabulary id. Special tokens keep their special
    id so binding supervision can distinguish ``<bos>`` from a content token.
    """

    if token_ids.shape != cue_types.shape:
        raise ValueError("token_ids and cue_types must share shape.")
    return token_ids.long()


def cue_span_for_position(
    positions: Tensor,
    *,
    cue_pos: Tensor,
    mask: Tensor,
) -> Tensor:
    """Build inclusive token spans ``[start, end]`` for each selected cue position.

    Args:
        positions: ``[batch, sequence]`` absolute or relative token positions.
        cue_pos: ``[batch, top_k]`` winning cue indices into the sequence axis.
        mask: ``[batch, sequence]`` active-token mask.

    Returns:
        Tensor shaped ``[batch, top_k, 2]``.
    """

    if positions.dim() != 2:
        raise ValueError("positions must have shape [batch, sequence].")
    if cue_pos.dim() != 2:
        raise ValueError("cue_pos must have shape [batch, top_k].")
    if mask.shape != positions.shape:
        raise ValueError("mask must match positions shape.")

    batch_size, top_k = cue_pos.shape
    starts = positions.gather(1, cue_pos.clamp_min(0))
    ends = starts.clone()

    pooled_rows = cue_pos.eq(POOLED_SOURCE_CUE_ID)
    if bool(pooled_rows.any()):
        seq_len = positions.shape[1]
        idx = torch.arange(seq_len, device=positions.device)
        idx_batch = idx.unsqueeze(0).expand(batch_size, -1)
        large = torch.full((batch_size, seq_len), seq_len, device=positions.device)
        small = torch.full((batch_size, seq_len), -1, device=positions.device)
        first = torch.where(mask, idx_batch, large).min(dim=1).values
        last = torch.where(mask, idx_batch, small).max(dim=1).values
        pooled_starts = first.unsqueeze(-1).expand(-1, top_k)
        pooled_ends = last.unsqueeze(-1).expand(-1, top_k)
        starts = torch.where(pooled_rows, pooled_starts, starts)
        ends = torch.where(pooled_rows, pooled_ends, ends)

    return torch.stack([starts, ends], dim=-1)


def build_trace_provenance(
    *,
    trace_ids: Tensor,
    source_cue_ids_per_trace: Tensor,
    token_ids: Tensor,
    positions: Tensor,
    mask: Tensor,
    cue_types_per_position: Tensor,
    routing_mode: str,
) -> TraceProvenance:
    """Gather provenance tensors for the selected top-k traces."""

    if trace_ids.dim() != 2:
        raise ValueError("trace_ids must have shape [batch, top_k].")
    if source_cue_ids_per_trace.dim() != 2:
        raise ValueError("source_cue_ids_per_trace must have shape [batch, num_traces].")

    batch_size, top_k = trace_ids.shape
    device = trace_ids.device

    if routing_mode == "pooled":
        source_cue_id = torch.full((batch_size, top_k), POOLED_SOURCE_CUE_ID, device=device, dtype=torch.long)
        source_token_id = torch.full((batch_size, top_k), POOLED_SOURCE_TOKEN_ID, device=device, dtype=torch.long)
        cue_type = torch.full((batch_size, top_k), CUE_TYPE_POOLED, device=device, dtype=torch.long)
        normalized = torch.full((batch_size, top_k), POOLED_NORMALIZED_CUE_ID, device=device, dtype=torch.long)
        source_span = cue_span_for_position(positions, cue_pos=source_cue_id, mask=mask)
        return TraceProvenance(
            source_cue_id=source_cue_id,
            source_token_id=source_token_id,
            source_span=source_span,
            cue_type=cue_type,
            normalized_cue_ids=normalized,
        )

    source_cue_id = source_cue_ids_per_trace.gather(1, trace_ids)
    safe_cue_id = source_cue_id.clamp_min(0)
    source_token_id = token_ids.gather(1, safe_cue_id)
    cue_type = cue_types_per_position.gather(1, safe_cue_id)
    normalized = normalize_cue_ids(source_token_id, cue_type)
    source_span = cue_span_for_position(positions, cue_pos=source_cue_id, mask=mask)

    return TraceProvenance(
        source_cue_id=source_cue_id,
        source_token_id=source_token_id,
        source_span=source_span,
        cue_type=cue_type,
        normalized_cue_ids=normalized,
    )


def format_cue_type_label(cue_type: int) -> str:
    return CUE_TYPE_LABELS.get(int(cue_type), f"type_{int(cue_type)}")


def preview_provenance_row(
    *,
    source_cue_id: int,
    source_token_id: int,
    source_span: tuple[int, int],
    cue_type: int,
    normalized_cue_id: int,
) -> str:
    span_text = f"{source_span[0]}-{source_span[1]}"
    return (
        f"cue={source_cue_id} token_id={source_token_id} "
        f"span={span_text} type={format_cue_type_label(cue_type)} norm_id={normalized_cue_id}"
    )


__all__ = [
    "CUE_TYPE_BOS",
    "CUE_TYPE_EOS",
    "CUE_TYPE_LABELS",
    "CUE_TYPE_MASK",
    "CUE_TYPE_PAD",
    "CUE_TYPE_POOLED",
    "CUE_TYPE_TOKEN",
    "CUE_TYPE_UNK",
    "POOLED_NORMALIZED_CUE_ID",
    "POOLED_SOURCE_CUE_ID",
    "POOLED_SOURCE_TOKEN_ID",
    "SpecialTokenIds",
    "TraceProvenance",
    "build_trace_provenance",
    "classify_cue_types",
    "cue_span_for_position",
    "format_cue_type_label",
    "normalize_cue_ids",
    "preview_provenance_row",
]
