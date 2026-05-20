"""Pair-derived basin support (Commit D2).

Maps bound trace pairs into basin pressure vectors so basins encode relationship
patterns, not isolated trace activations alone.
"""

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

LOGGER = logging.getLogger(__name__)


def _log_trace(logger: logging.Logger, enabled: bool, event: str, **fields: object) -> None:
    if not enabled:
        return
    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s%s", event, f" | {details}" if details else "")


def _preview_tensor(tensor: Tensor, *, limit: int = 6) -> str:
    flat = tensor.detach().cpu().reshape(-1)
    values = [f"{float(value):.4f}" for value in flat[:limit]]
    suffix = f", ... (+{flat.numel() - limit})" if flat.numel() > limit else ""
    return "[" + ", ".join(values) + suffix + "]"


@dataclass
class PairDerivedBasinSupportConfig:
    content_dim: int
    num_basins: int
    num_relations: int
    relation_embed_dim: int = 8
    hidden_dim: int | None = None
    trace: bool = False

    def __post_init__(self) -> None:
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.num_relations <= 0:
            raise ValueError("num_relations must be positive.")
        if self.relation_embed_dim <= 0:
            raise ValueError("relation_embed_dim must be positive.")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")

    @property
    def resolved_hidden_dim(self) -> int:
        return self.hidden_dim if self.hidden_dim is not None else max(self.content_dim * 2, 16)


class PairDerivedBasinSupport(nn.Module):
    """Project ``pair_ijr`` features into per-basin support vectors."""

    def __init__(
        self,
        config: PairDerivedBasinSupportConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.logger = logger
        self.relation_embeddings = nn.Embedding(config.num_relations, config.relation_embed_dim)
        pair_input_dim = config.content_dim * 3 + config.relation_embed_dim + 1
        hidden = config.resolved_hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(pair_input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.num_basins),
        )
        nn.init.normal_(self.relation_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.projection[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection[0].bias)
        nn.init.normal_(self.projection[2].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection[2].bias)

    def build_pair_features(
        self,
        src_content: Tensor,
        dst_content: Tensor,
        *,
        relation_index: Tensor,
        binding_strength: Tensor,
    ) -> Tensor:
        """Build ``pair_ijr = [trace_i, trace_j, trace_i*trace_j, relation_r, strength]``."""

        if src_content.shape != dst_content.shape:
            raise ValueError("src_content and dst_content must have the same shape.")
        if src_content.dim() != 3:
            raise ValueError("src_content must have shape [batch, edges, content_dim].")
        if src_content.shape[-1] != self.config.content_dim:
            raise ValueError("src_content width must match content_dim.")
        if relation_index.shape[:2] != src_content.shape[:2]:
            raise ValueError("relation_index must have shape [batch, edges].")
        if binding_strength.shape != relation_index.shape:
            raise ValueError("binding_strength must match relation_index shape.")

        hadamard = src_content * dst_content
        relation_ids = relation_index.long()
        if (relation_ids < 0).any() or (relation_ids >= self.config.num_relations).any():
            raise ValueError("relation_index must be in [0, num_relations).")
        relation_r = self.relation_embeddings(relation_ids)
        strength = binding_strength.unsqueeze(-1).to(dtype=src_content.dtype)

        return torch.cat([src_content, dst_content, hadamard, relation_r, strength], dim=-1)

    def edge_support(
        self,
        src_content: Tensor,
        dst_content: Tensor,
        *,
        relation_index: Tensor,
        binding_strength: Tensor,
    ) -> Tensor:
        """Return per-edge basin support with shape ``[batch, edges, num_basins]``."""

        pair_ijr = self.build_pair_features(
            src_content,
            dst_content,
            relation_index=relation_index,
            binding_strength=binding_strength,
        )
        return self.projection(pair_ijr)

    def forward(
        self,
        src_content: Tensor,
        dst_content: Tensor,
        *,
        relation_index: Tensor,
        binding_strength: Tensor,
    ) -> Tensor:
        """Aggregate weighted edge support into basin pressure ``[batch, num_basins]``."""

        edge_vectors = self.edge_support(
            src_content,
            dst_content,
            relation_index=relation_index,
            binding_strength=binding_strength,
        )
        aggregated = self.aggregate_edge_support(edge_vectors, binding_strength)
        _log_trace(
            self.logger,
            self.config.trace,
            "pair_basin_support.forward",
            batch=aggregated.shape[0],
            edges=src_content.shape[1],
            basins=aggregated.shape[-1],
            preview=_preview_tensor(aggregated),
        )
        return aggregated

    @staticmethod
    def aggregate_edge_support(edge_support: Tensor, binding_strength: Tensor) -> Tensor:
        if edge_support.dim() != 3:
            raise ValueError("edge_support must have shape [batch, edges, num_basins].")
        if binding_strength.shape != edge_support.shape[:2]:
            raise ValueError("binding_strength must have shape [batch, edges].")
        if edge_support.shape[1] == 0:
            return torch.zeros(
                edge_support.shape[0],
                edge_support.shape[2],
                device=edge_support.device,
                dtype=edge_support.dtype,
            )

        weights = binding_strength.clamp_min(0.0).unsqueeze(-1)
        return (edge_support * weights).sum(dim=1)


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect pair-derived basin support.")
    parser.add_argument(
        "command",
        nargs="?",
        default="inspect",
        choices=["inspect"],
        help="Inspection command (default: inspect).",
    )
    parser.add_argument("--content-dim", type=int, default=8)
    parser.add_argument("--num-basins", type=int, default=12)
    parser.add_argument("--num-relations", type=int, default=4)
    parser.add_argument("--num-edges", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    torch.manual_seed(args.seed)

    module = PairDerivedBasinSupport(
        PairDerivedBasinSupportConfig(
            content_dim=args.content_dim,
            num_basins=args.num_basins,
            num_relations=args.num_relations,
            trace=args.trace,
        )
    )
    module.eval()

    batch_size = 1
    src = torch.randn(batch_size, args.num_edges, args.content_dim)
    dst = torch.randn(batch_size, args.num_edges, args.content_dim)
    relation_index = torch.arange(args.num_edges, dtype=torch.long).unsqueeze(0) % args.num_relations
    strength = torch.linspace(0.2, 0.9, args.num_edges).unsqueeze(0)

    with torch.no_grad():
        edge = module.edge_support(src, dst, relation_index=relation_index, binding_strength=strength)
        aggregated = module(src, dst, relation_index=relation_index, binding_strength=strength)

    _safe_print(f"Pair feature width: {module.build_pair_features(src, dst, relation_index=relation_index, binding_strength=strength).shape[-1]}")
    _safe_print(f"Edge support shape: {list(edge.shape)}")
    _safe_print(f"Aggregated basin support shape: {list(aggregated.shape)}")
    _safe_print(f"Aggregated preview: {_preview_tensor(aggregated)}")


__all__ = ["PairDerivedBasinSupport", "PairDerivedBasinSupportConfig"]


if __name__ == "__main__":
    main()
