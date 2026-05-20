"""Learnable basin bank: attractor vectors and batch pressure state."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.basin.basin_state import BasinStateSpec, make_basin_state, validate_basin_state  # noqa: E402
from lmf.core.state.types import BasinState  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _log_trace(logger: logging.Logger, enabled: bool, event: str, **fields: object) -> None:
    if not enabled:
        return
    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s%s", event, f" | {details}" if details else "")


@dataclass
class BasinBankConfig:
    num_basins: int
    basin_dim: int
    init_scale: float = 0.02
    trace: bool = False

    def __post_init__(self) -> None:
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.basin_dim <= 0:
            raise ValueError("basin_dim must be positive.")
        if self.init_scale <= 0.0:
            raise ValueError("init_scale must be positive.")


class BasinBank(nn.Module):
    """Fixed-size bank of learnable basin attractor vectors."""

    def __init__(
        self,
        config: BasinBankConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.logger = logger
        self.vectors = nn.Parameter(torch.empty(config.num_basins, config.basin_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.vectors, mean=0.0, std=self.config.init_scale)

    @property
    def num_basins(self) -> int:
        return self.config.num_basins

    @property
    def basin_dim(self) -> int:
        return self.config.basin_dim

    def batch_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        pressures: Tensor | None = None,
    ) -> BasinState:
        """Create a validated per-batch basin state backed by this bank."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        vectors = self.vectors
        device = device if device is not None else vectors.device
        dtype = dtype if dtype is not None else vectors.dtype

        basin_state = make_basin_state(
            batch_size=batch_size,
            num_basins=self.num_basins,
            basin_dim=self.basin_dim,
            device=device,
            dtype=dtype,
            vectors=vectors,
            pressures=pressures,
        )
        validate_basin_state(
            basin_state,
            spec=BasinStateSpec(
                batch_size=batch_size,
                num_basins=self.num_basins,
                basin_dim=self.basin_dim,
            ),
        )
        _log_trace(
            self.logger,
            self.config.trace,
            "basin_bank.batch_state",
            batch=batch_size,
            basins=self.num_basins,
            dim=self.basin_dim,
        )
        return basin_state


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the learnable basin bank.")
    parser.add_argument(
        "command",
        nargs="?",
        default="inspect",
        choices=["inspect"],
    )
    parser.add_argument("--num-basins", type=int, default=16)
    parser.add_argument("--basin-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    torch.manual_seed(args.seed)

    bank = BasinBank(
        BasinBankConfig(
            num_basins=args.num_basins,
            basin_dim=args.basin_dim,
            trace=args.trace,
        )
    )
    state = bank.batch_state(args.batch_size)
    _safe_print(f"Basin bank vectors shape: {list(bank.vectors.shape)}")
    _safe_print(f"Batch pressures shape: {list(state.pressures.shape)}")
    _safe_print(f"Basin ids: {state.basin_ids.tolist() if state.basin_ids is not None else []}")


__all__ = ["BasinBank", "BasinBankConfig"]


if __name__ == "__main__":
    main()
