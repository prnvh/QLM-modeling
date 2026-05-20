"""Basin-family contrastive objective (Commit D3).

Contrastive loss acts on **basin attractor state**: settled pressures mixed through
learned ``basin_bank.vectors``. Family strings are weak sentence-level supervision,
not hand-assigned basin slot meanings.
"""

from __future__ import annotations

import argparse
import logging
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


def _log_trace(logger: logging.Logger, enabled: bool, event: str, **fields: object) -> None:
    if not enabled:
        return
    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s%s", event, f" | {details}" if details else "")


@dataclass
class BasinFamilyContrastiveConfig:
    num_basins: int
    basin_dim: int
    temperature: float = 0.1
    loss_weight: float = 0.05
    trace: bool = False

    def __post_init__(self) -> None:
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.basin_dim <= 0:
            raise ValueError("basin_dim must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.loss_weight < 0.0:
            raise ValueError("loss_weight must be non-negative.")


@dataclass(frozen=True)
class BasinFamilyContrastiveMetrics:
    """Diagnostics for family separation in attractor state space."""

    loss: float
    same_family_cosine_mean: float
    different_family_cosine_mean: float
    separation_margin: float
    num_pairs_same: int
    num_pairs_different: int
    readout_mode: str = "attractor"

    @property
    def separation_ok(self) -> bool:
        return self.separation_margin > 0.0


def compute_basin_attractor_state(
    basin_pressures: Tensor,
    basin_vectors: Tensor,
) -> Tensor:
    """Mix slot pressures through learnable attractor vectors.

    ``state = pressures @ vectors`` → shape ``[batch, basin_dim]``, L2-normalized.
    Gradients flow into both pressures (field loop) and vectors (basin bank).
    """

    if basin_pressures.dim() != 2:
        raise ValueError("basin_pressures must have shape [batch, num_basins].")
    if basin_vectors.dim() != 2:
        raise ValueError("basin_vectors must have shape [num_basins, basin_dim].")
    if basin_pressures.shape[1] != basin_vectors.shape[0]:
        raise ValueError("basin_pressures width must match basin_vectors rows.")
    if not torch.isfinite(basin_pressures).all():
        raise ValueError("basin_pressures contains non-finite values.")
    if not torch.isfinite(basin_vectors).all():
        raise ValueError("basin_vectors contains non-finite values.")

    state = basin_pressures @ basin_vectors
    return F.normalize(state, dim=-1, eps=1e-6)


def family_indices_tensor(
    family_ids: list[int],
    *,
    device: torch.device,
) -> Tensor:
    if not family_ids:
        raise ValueError("family_ids must not be empty.")
    if any(index < 0 for index in family_ids):
        raise ValueError("family_ids must be non-negative.")
    return torch.tensor(family_ids, device=device, dtype=torch.long)


def supervised_contrastive_loss(
    embeddings: Tensor,
    family_index: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Supervised contrastive loss (same family = positive, different = negative)."""

    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape [batch, basin_dim].")
    if family_index.dim() != 1 or family_index.shape[0] != embeddings.shape[0]:
        raise ValueError("family_index must have shape [batch].")
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    batch_size = embeddings.shape[0]
    device = embeddings.device
    normalized = F.normalize(embeddings, dim=-1, eps=1e-6)
    logits = torch.matmul(normalized, normalized.T) / temperature

    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    same_family = family_index.unsqueeze(0) == family_index.unsqueeze(1)
    positive_mask = same_family & ~self_mask

    losses: list[Tensor] = []
    for anchor in range(batch_size):
        positives = positive_mask[anchor]
        if not bool(positives.any()):
            continue
        anchor_logits = logits[anchor].masked_fill(self_mask[anchor], float("-inf"))
        positive_logits = anchor_logits.masked_fill(~positives, float("-inf"))
        numerator = torch.logsumexp(positive_logits, dim=0)
        denominator = torch.logsumexp(anchor_logits, dim=0)
        losses.append(-(numerator - denominator))

    if not losses:
        return embeddings.new_zeros(())
    return torch.stack(losses).mean()


def cosine_separation_metrics(
    embeddings: Tensor,
    family_index: Tensor,
) -> BasinFamilyContrastiveMetrics:
    """Compute same-family vs different-family cosine similarity diagnostics."""

    if embeddings.shape[0] < 2:
        return BasinFamilyContrastiveMetrics(
            loss=0.0,
            same_family_cosine_mean=0.0,
            different_family_cosine_mean=0.0,
            separation_margin=0.0,
            num_pairs_same=0,
            num_pairs_different=0,
        )

    normalized = F.normalize(embeddings.detach(), dim=-1, eps=1e-6)
    batch_size = normalized.shape[0]
    same_values: list[float] = []
    diff_values: list[float] = []

    for left in range(batch_size):
        for right in range(left + 1, batch_size):
            cosine = float(torch.dot(normalized[left], normalized[right]).item())
            if int(family_index[left].item()) == int(family_index[right].item()):
                same_values.append(cosine)
            else:
                diff_values.append(cosine)

    same_mean = sum(same_values) / len(same_values) if same_values else 0.0
    diff_mean = sum(diff_values) / len(diff_values) if diff_values else 0.0
    return BasinFamilyContrastiveMetrics(
        loss=0.0,
        same_family_cosine_mean=same_mean,
        different_family_cosine_mean=diff_mean,
        separation_margin=same_mean - diff_mean,
        num_pairs_same=len(same_values),
        num_pairs_different=len(diff_values),
    )


class BasinFamilyContrastive(nn.Module):
    """Attractor-state contrastive loss (no auxiliary embedding head)."""

    def __init__(
        self,
        config: BasinFamilyContrastiveConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.logger = logger

    def embed(
        self,
        basin_pressures: Tensor,
        basin_vectors: Tensor,
    ) -> Tensor:
        return compute_basin_attractor_state(basin_pressures, basin_vectors)

    def forward(
        self,
        basin_pressures: Tensor,
        basin_vectors: Tensor,
        family_index: Tensor,
    ) -> tuple[Tensor, BasinFamilyContrastiveMetrics]:
        state = self.embed(basin_pressures, basin_vectors)
        raw_loss = supervised_contrastive_loss(
            state,
            family_index,
            temperature=self.config.temperature,
        )
        weighted = raw_loss * self.config.loss_weight
        metrics = cosine_separation_metrics(state, family_index)
        metrics = BasinFamilyContrastiveMetrics(
            loss=float(weighted.detach().item()),
            same_family_cosine_mean=metrics.same_family_cosine_mean,
            different_family_cosine_mean=metrics.different_family_cosine_mean,
            separation_margin=metrics.separation_margin,
            num_pairs_same=metrics.num_pairs_same,
            num_pairs_different=metrics.num_pairs_different,
            readout_mode="attractor",
        )
        _log_trace(
            self.logger,
            self.config.trace,
            "basin_family_contrastive.forward",
            batch=state.shape[0],
            basin_dim=state.shape[1],
            loss=f"{metrics.loss:.6f}",
            same_cos=f"{metrics.same_family_cosine_mean:.4f}",
            diff_cos=f"{metrics.different_family_cosine_mean:.4f}",
            margin=f"{metrics.separation_margin:.4f}",
        )
        return weighted, metrics

    def metrics_only(
        self,
        basin_pressures: Tensor,
        basin_vectors: Tensor,
        family_index: Tensor,
    ) -> BasinFamilyContrastiveMetrics:
        state = self.embed(basin_pressures, basin_vectors)
        return cosine_separation_metrics(state, family_index)


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect basin-family contrastive readout.")
    parser.add_argument(
        "command",
        nargs="?",
        default="inspect",
        choices=["inspect"],
    )
    parser.add_argument("--num-basins", type=int, default=16)
    parser.add_argument("--basin-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    torch.manual_seed(args.seed)

    vectors = torch.randn(args.num_basins, args.basin_dim)
    module = BasinFamilyContrastive(
        BasinFamilyContrastiveConfig(
            num_basins=args.num_basins,
            basin_dim=args.basin_dim,
            trace=args.trace,
        )
    )
    module.eval()

    batch_size = max(args.batch_size, 2)
    pressures = torch.randn(batch_size, args.num_basins)
    family_index = torch.tensor([0, 0, 1, 1, 2, 2][:batch_size], dtype=torch.long)

    with torch.no_grad():
        loss, metrics = module(pressures, vectors, family_index)

    _safe_print(f"Readout: pressures @ basin_bank.vectors → [{batch_size}, {args.basin_dim}]")
    _safe_print(f"Weighted loss: {float(loss.item()):.6f}")
    _safe_print(
        "Metrics: "
        f"same_cos={metrics.same_family_cosine_mean:.4f} "
        f"diff_cos={metrics.different_family_cosine_mean:.4f} "
        f"margin={metrics.separation_margin:.4f}"
    )


__all__ = [
    "BasinFamilyContrastive",
    "BasinFamilyContrastiveConfig",
    "BasinFamilyContrastiveMetrics",
    "compute_basin_attractor_state",
    "cosine_separation_metrics",
    "family_indices_tensor",
    "supervised_contrastive_loss",
]


if __name__ == "__main__":
    main()
