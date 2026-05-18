"""Non-transformer cue encoder for Stage 1 input signals.

The encoder deliberately stays local: token ids become embeddings, a small
1-D convolution adds neighboring-token features, and a projection emits cue
vectors. It does not use dense token-token attention.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.input.cue_packet import CuePacket  # noqa: E402
from lmf.core.input.tokenizer import (  # noqa: E402
    SimpleTokenizer,
    Vocabulary,
    build_vocabulary_from_texts,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_TRACE_LIMIT = 8


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


def _preview_ints(values: Iterable[int], *, limit: int) -> str:
    if limit < 0:
        raise ValueError("integer preview limit must be non-negative.")
    items = list(values)
    shown = [str(value) for value in items[:limit]]
    if len(items) > limit:
        suffix = f"{', ' if shown else ''}... (+{len(items) - limit})"
    else:
        suffix = ""
    return "[" + ", ".join(shown) + suffix + "]"


@dataclass
class CueEncoderConfig:
    """Configuration for the local cue encoder."""

    vocab_size: int
    cue_dim: int
    embedding_dim: int | None = None
    hidden_dim: int | None = None
    pad_id: int = 0
    window_size: int = 3
    dropout: float = 0.0
    trace: bool = False
    trace_limit: int = DEFAULT_TRACE_LIMIT

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.cue_dim <= 0:
            raise ValueError("cue_dim must be positive.")
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive when provided.")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")
        if self.pad_id < 0 or self.pad_id >= self.vocab_size:
            raise ValueError("pad_id must be within the vocabulary.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.window_size % 2 == 0:
            raise ValueError("window_size must be odd so cue length is preserved.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if self.trace_limit < 0:
            raise ValueError("trace_limit must be non-negative.")

    @property
    def resolved_embedding_dim(self) -> int:
        return self.embedding_dim if self.embedding_dim is not None else self.cue_dim

    @property
    def resolved_hidden_dim(self) -> int:
        return self.hidden_dim if self.hidden_dim is not None else self.cue_dim


class CueEncoder(nn.Module):
    """Embed token ids into local cue vectors and return a CuePacket."""

    def __init__(
        self,
        config: CueEncoderConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.logger = logger
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.resolved_embedding_dim,
            padding_idx=config.pad_id,
        )
        self.local_conv = nn.Conv1d(
            config.resolved_embedding_dim,
            config.resolved_hidden_dim,
            kernel_size=config.window_size,
            padding=config.window_size // 2,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(config.resolved_hidden_dim, config.cue_dim)
        self.layer_norm = nn.LayerNorm(config.cue_dim)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> CuePacket:
        """Encode ids with shape [batch, sequence] or [sequence]."""

        input_ids = self._normalize_input_ids(input_ids)
        mask = self._normalize_mask(input_ids, attention_mask)
        positions = self._normalize_positions(input_ids, positions)

        embeddings = self.embedding(input_ids)
        embeddings = embeddings * mask.unsqueeze(-1).to(dtype=embeddings.dtype)

        local_features = self.local_conv(embeddings.transpose(1, 2)).transpose(1, 2)
        local_features = self.activation(local_features)
        local_features = self.dropout(local_features)
        cues = self.output_projection(local_features)
        cues = self.layer_norm(cues)
        cues = cues * mask.unsqueeze(-1).to(dtype=cues.dtype)

        token_counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=cues.dtype)
        pooled = cues.sum(dim=1) / token_counts

        _log_trace(
            self.logger,
            self.config.trace,
            "cue_encoder.forward",
            input_shape=list(input_ids.shape),
            cue_shape=list(cues.shape),
            active_tokens=int(mask.sum().item()),
            ids=_preview_ints(input_ids.reshape(-1).tolist(), limit=self.config.trace_limit),
            pooled=_preview_tensor(pooled, limit=self.config.trace_limit),
        )

        return CuePacket(cues=cues, mask=mask, positions=positions, pooled=pooled)

    def _normalize_input_ids(self, input_ids: Tensor) -> Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [sequence] or [batch, sequence].")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be an integer tensor.")
        input_ids = input_ids.long()
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty.")
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0 or max_id >= self.config.vocab_size:
            raise ValueError("input_ids contain ids outside the configured vocabulary.")
        return input_ids

    def _normalize_mask(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        if attention_mask is None:
            return input_ids.ne(self.config.pad_id)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
        return attention_mask.bool()

    def _normalize_positions(self, input_ids: Tensor, positions: Tensor | None) -> Tensor:
        if positions is None:
            seq_len = input_ids.shape[1]
            return torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        if positions.shape != input_ids.shape:
            raise ValueError("positions must match input_ids shape.")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must be an integer tensor.")
        return positions.long()


def encode_text(
    text: str,
    *,
    cue_dim: int = 16,
    embedding_dim: int | None = None,
    hidden_dim: int | None = None,
    window_size: int = 3,
    add_bos: bool = True,
    add_eos: bool = True,
    seed: int = 7,
    trace: bool = False,
) -> tuple[Vocabulary, list[int], CuePacket]:
    """Build a temporary vocab from text and encode it for CLI inspection."""

    torch.manual_seed(seed)
    vocab = build_vocabulary_from_texts([text])
    tokenizer = SimpleTokenizer(vocab=vocab)
    ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    encoder = CueEncoder(
        CueEncoderConfig(
            vocab_size=len(vocab),
            cue_dim=cue_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pad_id=vocab.pad_id,
            window_size=window_size,
            trace=trace,
        )
    )
    encoder.eval()
    with torch.no_grad():
        packet = encoder(torch.tensor(ids, dtype=torch.long))
    return vocab, ids, packet


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
    parser = argparse.ArgumentParser(description="Encode text into local cue vectors.")
    parser.add_argument(
        "text",
        nargs="+",
        help='Text to encode. You may optionally start with the word "text".',
    )
    parser.add_argument("--cue-dim", type=int, default=16, help="Cue vector width.")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Token embedding width.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Local convolution hidden width.")
    parser.add_argument("--window-size", type=int, default=3, help="Odd local convolution window size.")
    parser.add_argument("--seed", type=int, default=7, help="Initialization seed for reproducible inspection.")
    parser.add_argument("--no-bos", action="store_true", help="Do not prepend <bos>.")
    parser.add_argument("--no-eos", action="store_true", help="Do not append <eos>.")
    parser.add_argument("--trace", action="store_true", help="Print human-readable cue encoder trace logs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)
    vocab, ids, packet = encode_text(
        text,
        cue_dim=args.cue_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        window_size=args.window_size,
        add_bos=not args.no_bos,
        add_eos=not args.no_eos,
        seed=args.seed,
        trace=args.trace,
    )

    first_row = packet.cues[0, 0] if packet.cues.shape[1] else torch.empty(0)
    _safe_print(f"Input: {text}")
    _safe_print(f"Vocab size: {len(vocab)}")
    _safe_print(f"Token IDs: {ids}")
    _safe_print(f"Cue shape: {list(packet.cues.shape)}")
    _safe_print(f"Mask: {packet.mask.int().tolist() if packet.mask is not None else None}")
    _safe_print(f"Positions: {packet.positions.tolist() if packet.positions is not None else None}")
    _safe_print(f"First cue: {_preview_tensor(first_row, limit=min(args.cue_dim, 8))}")
    _safe_print(f"Pooled cue: {_preview_tensor(packet.pooled[0], limit=min(args.cue_dim, 8))}")


__all__ = [
    "CueEncoder",
    "CueEncoderConfig",
    "CuePacket",
    "encode_text",
]


if __name__ == "__main__":
    main()
