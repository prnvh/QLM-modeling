"""Learnable distributed memory field (DMF) trace bank.

Each trace slot stores a routing key, a content vector, an activation threshold,
and a decay rate. Sparse routing and active-region construction are implemented
in ``trace_router.py`` and ``active_region.py``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _preview_ints(values: list[int], *, limit: int) -> str:
    if limit < 0:
        raise ValueError("integer preview limit must be non-negative.")
    shown = [str(value) for value in values[:limit]]
    if len(values) > limit:
        suffix = f"{', ' if shown else ''}... (+{len(values) - limit})"
    else:
        suffix = ""
    return "[" + ", ".join(shown) + suffix + "]"


@dataclass
class TraceBankConfig:
    """Configuration for the learnable trace bank."""

    num_traces: int
    key_dim: int
    content_dim: int
    init_scale: float = 0.02
    initial_threshold: float = 0.1
    initial_decay: float = 0.05
    trace: bool = False
    trace_limit: int = DEFAULT_TRACE_LIMIT

    def __post_init__(self) -> None:
        if self.num_traces <= 0:
            raise ValueError("num_traces must be positive.")
        if self.key_dim <= 0:
            raise ValueError("key_dim must be positive.")
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.init_scale <= 0.0:
            raise ValueError("init_scale must be positive.")
        if self.initial_threshold <= 0.0:
            raise ValueError("initial_threshold must be positive.")
        if not 0.0 < self.initial_decay < 1.0:
            raise ValueError("initial_decay must be in the open interval (0, 1).")
        if self.trace_limit < 0:
            raise ValueError("trace_limit must be non-negative.")


@dataclass
class TraceBankState:
    """Materialized trace parameters for inspection or downstream routing."""

    keys: Tensor
    content: Tensor
    threshold: Tensor
    decay: Tensor

    @property
    def num_traces(self) -> int:
        return int(self.keys.shape[0])


class TraceBank(nn.Module):
    """Fixed-size bank of learnable memory traces."""

    def __init__(
        self,
        config: TraceBankConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.logger = logger

        self.keys = nn.Parameter(
            torch.empty(config.num_traces, config.key_dim),
        )
        self.content = nn.Parameter(
            torch.empty(config.num_traces, config.content_dim),
        )
        threshold_raw = self._inverse_softplus(config.initial_threshold)
        self.threshold = nn.Parameter(torch.full((config.num_traces,), threshold_raw))
        decay_raw = self._inverse_sigmoid(config.initial_decay)
        self.decay = nn.Parameter(torch.full((config.num_traces,), decay_raw))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trace vectors and per-trace scalars."""

        nn.init.normal_(self.keys, mean=0.0, std=self.config.init_scale)
        nn.init.normal_(self.content, mean=0.0, std=self.config.init_scale)
        threshold_raw = self._inverse_softplus(self.config.initial_threshold)
        decay_raw = self._inverse_sigmoid(self.config.initial_decay)
        with torch.no_grad():
            self.threshold.fill_(threshold_raw)
            self.decay.fill_(decay_raw)

    def forward(self, trace_ids: Tensor | None = None) -> TraceBankState:
        """Return the full bank or a gathered subset by trace id."""

        return self.state(trace_ids=trace_ids)

    def state(self, *, trace_ids: Tensor | None = None) -> TraceBankState:
        """Expose keys, content, threshold, and decay with stable scalar ranges."""

        keys = self.keys
        content = self.content
        threshold = self._positive_threshold(self.threshold)
        decay = torch.sigmoid(self.decay)

        if trace_ids is not None:
            trace_ids = self._normalize_trace_ids(trace_ids)
            keys = keys.index_select(0, trace_ids)
            content = content.index_select(0, trace_ids)
            threshold = threshold.index_select(0, trace_ids)
            decay = decay.index_select(0, trace_ids)

        _log_trace(
            self.logger,
            self.config.trace,
            "trace_bank.state",
            num_traces=int(keys.shape[0]),
            key_shape=list(keys.shape),
            content_shape=list(content.shape),
            threshold=_preview_tensor(threshold, limit=self.config.trace_limit),
            decay=_preview_tensor(decay, limit=self.config.trace_limit),
            trace_ids=(
                _preview_ints(trace_ids.detach().cpu().tolist(), limit=self.config.trace_limit)
                if trace_ids is not None
                else "all"
            ),
        )

        return TraceBankState(
            keys=keys,
            content=content,
            threshold=threshold,
            decay=decay,
        )

    def _normalize_trace_ids(self, trace_ids: Tensor) -> Tensor:
        if trace_ids.dim() != 1:
            raise ValueError("trace_ids must have shape [num_selected].")
        if trace_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("trace_ids must be an integer tensor.")
        if trace_ids.numel() == 0:
            raise ValueError("trace_ids must not be empty.")
        trace_ids = trace_ids.long()
        min_id = int(trace_ids.min().item())
        max_id = int(trace_ids.max().item())
        if min_id < 0 or max_id >= self.config.num_traces:
            raise ValueError("trace_ids contain ids outside the configured trace bank.")
        return trace_ids

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value <= 0.0:
            raise ValueError("value for inverse softplus must be positive.")
        return math.log(math.expm1(value))

    @staticmethod
    def _inverse_sigmoid(value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("value for inverse sigmoid must be in (0, 1).")
        return math.log(value / (1.0 - value))

    @staticmethod
    def _positive_threshold(raw_threshold: Tensor) -> Tensor:
        return F.softplus(raw_threshold) + 1e-6


def build_trace_bank(
    *,
    num_traces: int = 1024,
    key_dim: int = 32,
    content_dim: int = 32,
    seed: int = 7,
    trace: bool = False,
) -> TraceBank:
    """Construct a trace bank with reproducible initialization for inspection."""

    torch.manual_seed(seed)
    bank = TraceBank(
        TraceBankConfig(
            num_traces=num_traces,
            key_dim=key_dim,
            content_dim=content_dim,
            trace=trace,
        )
    )
    bank.eval()
    return bank


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
    parser = argparse.ArgumentParser(description="Inspect the learnable DMF trace bank.")
    parser.add_argument(
        "command",
        nargs="?",
        default="inspect",
        choices=("inspect",),
        help="Subcommand (default: inspect).",
    )
    parser.add_argument("--num-traces", type=int, default=16, help="Number of trace slots.")
    parser.add_argument("--key-dim", type=int, default=8, help="Trace key vector width.")
    parser.add_argument("--content-dim", type=int, default=8, help="Trace content vector width.")
    parser.add_argument("--seed", type=int, default=7, help="Initialization seed.")
    parser.add_argument(
        "--sample-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional trace ids to gather for a subset view.",
    )
    parser.add_argument("--trace", action="store_true", help="Print human-readable trace bank logs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    bank = build_trace_bank(
        num_traces=args.num_traces,
        key_dim=args.key_dim,
        content_dim=args.content_dim,
        seed=args.seed,
        trace=args.trace,
    )

    with torch.no_grad():
        full = bank.state()
        if args.sample_ids is not None:
            subset = bank.state(trace_ids=torch.tensor(args.sample_ids, dtype=torch.long))
        else:
            subset = None

    _safe_print(f"Num traces: {full.num_traces}")
    _safe_print(f"Keys shape: {list(full.keys.shape)}")
    _safe_print(f"Content shape: {list(full.content.shape)}")
    _safe_print(f"Threshold shape: {list(full.threshold.shape)}")
    _safe_print(f"Decay shape: {list(full.decay.shape)}")
    _safe_print(f"First key: {_preview_tensor(full.keys[0], limit=min(args.key_dim, 8))}")
    _safe_print(f"First content: {_preview_tensor(full.content[0], limit=min(args.content_dim, 8))}")
    _safe_print(
        f"First threshold/decay: {full.threshold[0].item():.4f} / {full.decay[0].item():.4f}"
    )

    if subset is not None:
        _safe_print(f"Subset ids: {args.sample_ids}")
        _safe_print(f"Subset keys shape: {list(subset.keys.shape)}")
        _safe_print(f"Subset threshold: {_preview_tensor(subset.threshold, limit=8)}")


__all__ = [
    "TraceBank",
    "TraceBankConfig",
    "TraceBankState",
    "build_trace_bank",
]


if __name__ == "__main__":
    main()
