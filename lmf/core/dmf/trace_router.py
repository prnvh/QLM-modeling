"""Sparse cue-to-trace router for the DMF."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.dmf.cue_provenance import (  # noqa: E402
    SpecialTokenIds,
    TraceProvenance,
    build_trace_provenance,
    classify_cue_types,
    preview_provenance_row,
)
from lmf.core.dmf.sparsity import SparsityReport, topk_trace_routing  # noqa: E402
from lmf.core.support.human_report import (  # noqa: E402
    PipelineInspection,
    format_routing_section,
    resolve_source_tokens,
    tensors_to_ranked_lists,
)
from lmf.core.dmf.trace_bank import TraceBank, build_trace_bank  # noqa: E402
from lmf.core.input.cue_encoder import CueEncoder, CueEncoderConfig, encode_text  # noqa: E402
from lmf.core.input.cue_packet import CuePacket  # noqa: E402
from lmf.core.input.tokenizer import inspect_text  # noqa: E402
from lmf.core.state.types import ActiveRegion  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_TRACE_LIMIT = 8
RoutingMode = Literal["pooled", "max_token"]
ScoreMode = Literal["dot", "cosine"]


def _log_trace(logger: logging.Logger, enabled: bool, event: str, **fields: object) -> None:
    if not enabled:
        return
    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s%s", event, f" | {details}" if details else "")


def _preview_tensor(tensor: Tensor, *, limit: int) -> str:
    if limit < 0:
        raise ValueError("tensor preview limit must be non-negative.")
    flat = tensor.detach().cpu().reshape(-1)
    values = [float(value) for value in flat[:limit]]
    shown = [f"{value:.4f}" for value in values]
    if flat.numel() > limit:
        suffix = f"{', ' if shown else ''}... (+{flat.numel() - limit})"
    else:
        suffix = ""
    return "[" + ", ".join(shown) + suffix + "]"


def _preview_int_rows(values: Tensor, *, limit: int) -> str:
    if limit < 0:
        raise ValueError("integer preview limit must be non-negative.")
    rows = values.detach().cpu().tolist()
    shown_rows = rows[:limit]
    parts = ["[" + ", ".join(str(int(value)) for value in row) + "]" for row in shown_rows]
    if len(rows) > limit:
        suffix = f"{', ' if parts else ''}... (+{len(rows) - limit} rows)"
    else:
        suffix = ""
    return "[" + ", ".join(parts) + suffix + "]"


def _preview_provenance_rows(provenance: TraceProvenance, *, limit: int) -> str:
    if limit < 0:
        raise ValueError("provenance preview limit must be non-negative.")
    batch_size, top_k = provenance.source_cue_id.shape
    if batch_size == 0 or top_k == 0:
        return "[]"
    rows: list[str] = []
    count = min(limit, top_k)
    for index in range(count):
        span = provenance.source_span[0, index].detach().cpu().tolist()
        rows.append(
            preview_provenance_row(
                source_cue_id=int(provenance.source_cue_id[0, index].item()),
                source_token_id=int(provenance.source_token_id[0, index].item()),
                source_span=(int(span[0]), int(span[1])),
                cue_type=int(provenance.cue_type[0, index].item()),
                normalized_cue_id=int(provenance.normalized_cue_ids[0, index].item()),
            )
        )
    if top_k > count:
        rows.append(f"... (+{top_k - count})")
    return "[" + "; ".join(rows) + "]"


@dataclass
class TraceRouterConfig:
    """Configuration for sparse cue-to-trace routing."""

    top_k: int
    routing_mode: RoutingMode = "max_token"
    score_mode: ScoreMode = "dot"
    trace: bool = False
    trace_limit: int = DEFAULT_TRACE_LIMIT

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.routing_mode not in ("pooled", "max_token"):
            raise ValueError("routing_mode must be 'pooled' or 'max_token'.")
        if self.score_mode not in ("dot", "cosine"):
            raise ValueError("score_mode must be 'dot' or 'cosine'.")
        if self.trace_limit < 0:
            raise ValueError("trace_limit must be non-negative.")


@dataclass
class RoutingResult:
    """Sparse trace selection for one batch of cues."""

    trace_ids: Tensor
    scores: Tensor
    batch_size: int
    top_k: int
    num_traces: int
    sparsity: SparsityReport
    source_cue_id: Tensor
    source_token_id: Tensor
    source_span: Tensor
    cue_type: Tensor
    normalized_cue_ids: Tensor

    def __post_init__(self) -> None:
        if self.trace_ids.shape != self.scores.shape:
            raise ValueError("trace_ids and scores must have the same shape.")
        if self.trace_ids.dim() != 2:
            raise ValueError("trace_ids must have shape [batch, top_k].")
        TraceProvenance(
            source_cue_id=self.source_cue_id,
            source_token_id=self.source_token_id,
            source_span=self.source_span,
            cue_type=self.cue_type,
            normalized_cue_ids=self.normalized_cue_ids,
        )


class TraceRouter(nn.Module):
    """Route cues to a small top-k subset of learnable traces."""

    def __init__(
        self,
        config: TraceRouterConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.logger = logger

    def forward(self, cue_packet: CuePacket, trace_bank: TraceBank) -> RoutingResult:
        """Score cues against trace keys and return the top-k trace ids."""

        cues = self._normalize_cues(cue_packet.cues)
        mask = self._normalize_mask(cue_packet, cues)
        bank_state = trace_bank.state()

        if cues.shape[-1] != bank_state.keys.shape[-1]:
            raise ValueError("cue width must match trace key width.")

        scores, source_cue_ids_per_trace = self._batch_trace_scores(
            cues=cues,
            mask=mask,
            keys=bank_state.keys,
            threshold=bank_state.threshold,
            pooled=cue_packet.pooled,
        )
        trace_ids, top_scores = topk_trace_routing(scores, top_k=self.config.top_k)
        provenance = self._build_provenance(
            cue_packet=cue_packet,
            cues=cues,
            mask=mask,
            trace_ids=trace_ids,
            source_cue_ids_per_trace=source_cue_ids_per_trace,
        )
        sparsity = SparsityReport.from_routing(
            batch_size=scores.shape[0],
            num_traces=bank_state.num_traces,
            active_traces=trace_ids.shape[-1],
        )

        _log_trace(
            self.logger,
            self.config.trace,
            "trace_router.forward",
            batch_size=scores.shape[0],
            num_traces=bank_state.num_traces,
            top_k=trace_ids.shape[-1],
            active_fraction=f"{sparsity.active_fraction:.4f}",
            trace_ids=_preview_int_rows(trace_ids, limit=self.config.trace_limit),
            scores=_preview_tensor(top_scores, limit=self.config.trace_limit),
            provenance=_preview_provenance_rows(provenance, limit=self.config.trace_limit),
        )

        return RoutingResult(
            trace_ids=trace_ids,
            scores=top_scores,
            batch_size=scores.shape[0],
            top_k=trace_ids.shape[-1],
            num_traces=bank_state.num_traces,
            sparsity=sparsity,
            source_cue_id=provenance.source_cue_id,
            source_token_id=provenance.source_token_id,
            source_span=provenance.source_span,
            cue_type=provenance.cue_type,
            normalized_cue_ids=provenance.normalized_cue_ids,
        )

    def _batch_trace_scores(
        self,
        *,
        cues: Tensor,
        mask: Tensor,
        keys: Tensor,
        threshold: Tensor,
        pooled: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.config.routing_mode == "pooled":
            cue_vectors = self._resolve_pooled_cues(cues=cues, mask=mask, pooled=pooled)
            token_scores = self._pairwise_scores(cue_vectors, keys)
            scores = token_scores
            source_cue_ids_per_trace = None
        else:
            token_scores = self._pairwise_scores(cues, keys)
            token_scores = token_scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            scores, source_cue_ids_per_trace = token_scores.max(dim=1)

        return scores - threshold.unsqueeze(0), source_cue_ids_per_trace

    def _build_provenance(
        self,
        *,
        cue_packet: CuePacket,
        cues: Tensor,
        mask: Tensor,
        trace_ids: Tensor,
        source_cue_ids_per_trace: Tensor | None,
    ) -> TraceProvenance:
        token_ids = self._normalize_token_ids(cue_packet, cues)
        positions = self._normalize_positions(cue_packet, cues)
        special = self._resolve_special_token_ids(cue_packet)
        cue_types_per_position = classify_cue_types(token_ids, special)

        if self.config.routing_mode == "pooled":
            return build_trace_provenance(
                trace_ids=trace_ids,
                source_cue_ids_per_trace=trace_ids.new_zeros(trace_ids.shape[0], trace_ids.shape[1]),
                token_ids=token_ids,
                positions=positions,
                mask=mask,
                cue_types_per_position=cue_types_per_position,
                routing_mode="pooled",
            )

        if source_cue_ids_per_trace is None:
            raise RuntimeError("max_token routing must provide per-trace source cue ids.")

        return build_trace_provenance(
            trace_ids=trace_ids,
            source_cue_ids_per_trace=source_cue_ids_per_trace,
            token_ids=token_ids,
            positions=positions,
            mask=mask,
            cue_types_per_position=cue_types_per_position,
            routing_mode="max_token",
        )

    def _normalize_token_ids(self, cue_packet: CuePacket, cues: Tensor) -> Tensor:
        if cue_packet.token_ids is None:
            raise ValueError("cue_packet.token_ids is required to attach trace provenance.")
        token_ids = cue_packet.token_ids
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.shape != cues.shape[:2]:
            raise ValueError("cue_packet.token_ids must match cue batch/sequence shape.")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("cue_packet.token_ids must be an integer tensor.")
        return token_ids.long()

    def _normalize_positions(self, cue_packet: CuePacket, cues: Tensor) -> Tensor:
        if cue_packet.positions is not None:
            positions = cue_packet.positions
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
            if positions.shape != cues.shape[:2]:
                raise ValueError("cue_packet.positions must match cue batch/sequence shape.")
            return positions.long()

        seq_len = cues.shape[1]
        return torch.arange(seq_len, device=cues.device).unsqueeze(0).expand(cues.shape[0], -1)

    def _resolve_special_token_ids(self, cue_packet: CuePacket) -> SpecialTokenIds:
        if cue_packet.special_token_ids is not None:
            pad_id, unk_id, mask_id, bos_id, eos_id = cue_packet.special_token_ids
            return SpecialTokenIds(
                pad_id=pad_id,
                unk_id=unk_id,
                mask_id=mask_id,
                bos_id=bos_id,
                eos_id=eos_id,
            )
        return SpecialTokenIds(pad_id=0, unk_id=1, mask_id=2, bos_id=3, eos_id=4)

    def _resolve_pooled_cues(
        self,
        *,
        cues: Tensor,
        mask: Tensor,
        pooled: Tensor | None,
    ) -> Tensor:
        if pooled is not None:
            if pooled.dim() == 1:
                pooled = pooled.unsqueeze(0)
            if pooled.shape != (cues.shape[0], cues.shape[-1]):
                raise ValueError("pooled cues must have shape [batch, cue_dim].")
            return pooled

        counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=cues.dtype)
        return cues.sum(dim=1) / counts

    def _pairwise_scores(self, cues: Tensor, keys: Tensor) -> Tensor:
        if self.config.score_mode == "cosine":
            cues = F.normalize(cues, dim=-1)
            keys = F.normalize(keys, dim=-1)
        if cues.dim() == 2:
            return cues @ keys.transpose(0, 1)
        return torch.einsum("bsd,nd->bsn", cues, keys)

    def _normalize_cues(self, cues: Tensor) -> Tensor:
        if cues.dim() == 2:
            cues = cues.unsqueeze(0)
        elif cues.dim() != 3:
            raise ValueError("cues must have shape [batch, sequence, cue_dim] or [sequence, cue_dim].")
        if cues.numel() == 0:
            raise ValueError("cues must not be empty.")
        if not torch.isfinite(cues).all():
            raise ValueError("cues must be finite.")
        return cues

    def _normalize_mask(self, cue_packet: CuePacket, cues: Tensor) -> Tensor:
        if cue_packet.mask is None:
            return torch.ones(cues.shape[:2], dtype=torch.bool, device=cues.device)
        mask = cue_packet.mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != cues.shape[:2]:
            raise ValueError("cue mask must match cue batch/sequence shape.")
        mask = mask.bool()
        if not mask.any():
            raise ValueError("cue mask must include at least one active position.")
        return mask


def route_text(
    text: str,
    *,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    routing_mode: RoutingMode = "max_token",
    score_mode: ScoreMode = "dot",
    seed: int = 7,
    trace: bool = False,
) -> tuple[CuePacket, RoutingResult, ActiveRegion]:
    """Encode text, route cues into a trace bank, and build an active region."""

    vocab, _ids, cue_packet = encode_text(text, cue_dim=cue_dim, seed=seed, trace=trace)
    bank = build_trace_bank(
        num_traces=num_traces,
        key_dim=cue_dim,
        content_dim=cue_dim,
        seed=seed,
        trace=trace,
    )
    router = TraceRouter(
        TraceRouterConfig(
            top_k=top_k,
            routing_mode=routing_mode,
            score_mode=score_mode,
            trace=trace,
        )
    )
    from lmf.core.dmf.active_region import build_active_region

    router.eval()
    with torch.no_grad():
        routing = router(cue_packet, bank)
        active_region = build_active_region(
            trace_bank=bank,
            routing=routing,
            cue_packet=cue_packet,
        )
    _ = vocab
    return cue_packet, routing, active_region


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        force=True,
    )


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route text cues into a sparse active trace region.")
    parser.add_argument(
        "text",
        nargs="+",
        help='Text to route. You may optionally start with the word "text".',
    )
    parser.add_argument("--num-traces", type=int, default=64, help="Trace bank size.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of active traces per batch row.")
    parser.add_argument("--cue-dim", type=int, default=16, help="Cue and trace key width.")
    parser.add_argument(
        "--routing-mode",
        choices=("pooled", "max_token"),
        default="max_token",
        help="Use pooled cues or max-over-token cue scores.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("dot", "cosine"),
        default="dot",
        help="Similarity function for cue-to-key comparison.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Initialization seed.")
    parser.add_argument("--trace", action="store_true", help="Print routing debug logs to stderr.")
    parser.add_argument(
        "--numbers",
        action="store_true",
        help="Print raw tensor-style output instead of the readable table.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)

    _cue_packet, routing, active_region = route_text(
        text,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        routing_mode=args.routing_mode,
        score_mode=args.score_mode,
        seed=args.seed,
        trace=args.trace,
    )

    if args.numbers:
        _safe_print(f"Input: {text}")
        _safe_print(f"Active trace ids: {routing.trace_ids.tolist()}")
        _safe_print(f"Active scores: {_preview_tensor(routing.scores, limit=min(args.top_k, 8))}")
        _safe_print(f"Sparsity: {routing.sparsity.active_traces}/{routing.sparsity.num_traces}")
        _safe_print(f"Trace content shape: {list(active_region.trace_content.shape)}")
        _safe_print(f"Trace amp: {_preview_tensor(active_region.trace_amp, limit=min(args.top_k, 8))}")
        _safe_print(f"Cue drive: {_preview_tensor(active_region.cue_drive, limit=min(args.top_k, 8))}")
        return

    trace_ids, match_scores, activations, cue_drives, source_cue_ids, source_token_ids, source_spans, cue_types, normalized = (
        tensors_to_ranked_lists(
            routing.trace_ids,
            routing.scores,
            active_region.trace_amp,
            active_region.cue_drive,
            source_cue_id=routing.source_cue_id,
            source_token_id=routing.source_token_id,
            source_span=routing.source_span,
            cue_type=routing.cue_type,
            normalized_cue_ids=routing.normalized_cue_ids,
        )
    )
    token_report = inspect_text(text)
    token_rows = list(token_report["token_ids"])  # type: ignore[arg-type]
    source_tokens = resolve_source_tokens(token_rows, source_cue_ids, cue_types)
    report = format_routing_section(
        PipelineInspection(
            input_text=text,
            tokens=list(token_report["tokens"]),  # type: ignore[arg-type]
            token_rows=token_rows,
            routing_mode=args.routing_mode,
            score_mode=args.score_mode,
            num_traces=args.num_traces,
            top_k=args.top_k,
            trace_ids=trace_ids,
            match_scores=match_scores,
            activations_before=activations,
            cue_drives=cue_drives,
            source_cue_ids=source_cue_ids,
            source_token_ids=source_token_ids,
            source_tokens=source_tokens,
            source_spans=source_spans,
            cue_types=cue_types,
            normalized_cue_ids=normalized,
        )
    )
    _safe_print(report)


__all__ = [
    "RoutingResult",
    "TraceRouter",
    "TraceRouterConfig",
    "route_text",
]


if __name__ == "__main__":
    main()
