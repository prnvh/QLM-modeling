"""Cognitive-state readout channels (Commit E1).

Each channel is a distinct latent projection from ``CognitiveState`` — not a raw cue
shortcut and not dense token-token attention. Binding readout uses **soft bound-pair**
features gathered from sparse edges (same relational substrate as D2 basins).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.basin.basin_family_contrastive import compute_basin_attractor_state  # noqa: E402
from lmf.core.basin.binding_edges import gather_binding_edge_batch  # noqa: E402
from lmf.core.dmf.trace_router import route_text  # noqa: E402
from lmf.core.field.context_op import ContextOp, ContextOpConfig  # noqa: E402
from lmf.core.field.loop import FieldLoop, FieldLoopConfig  # noqa: E402
from lmf.core.state.cognitive_state import (  # noqa: E402
    build_cognitive_state_from_field_loop,
    require_context_summary,
)
from lmf.core.state.types import CognitiveState  # noqa: E402

LOGGER = logging.getLogger(__name__)

READOUT_CHANNEL_NAMES: tuple[str, ...] = (
    "trace",
    "binding",
    "basin",
    "interference",
    "lucidity",
    "context",
)


def _log_trace(logger: logging.Logger, enabled: bool, event: str, **fields: object) -> None:
    if not enabled:
        return
    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s%s", event, f" | {details}" if details else "")


def _preview_tensor(tensor: Tensor, *, limit: int = 4) -> str:
    flat = tensor.detach().cpu().reshape(-1)
    values = [f"{float(value):.4f}" for value in flat[:limit]]
    suffix = f", ... (+{flat.numel() - limit})" if flat.numel() > limit else ""
    return "[" + ", ".join(values) + suffix + "]"


@dataclass
class ReadoutChannelsConfig:
    content_dim: int
    channel_dim: int
    num_relations: int = 4
    relation_embed_dim: int = 8
    hidden_dim: int | None = None
    trace: bool = False

    def __post_init__(self) -> None:
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.channel_dim <= 0:
            raise ValueError("channel_dim must be positive.")
        if self.num_relations <= 0:
            raise ValueError("num_relations must be positive.")
        if self.relation_embed_dim <= 0:
            raise ValueError("relation_embed_dim must be positive.")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")

    @property
    def resolved_hidden_dim(self) -> int:
        return self.hidden_dim if self.hidden_dim is not None else max(self.channel_dim, self.content_dim)

    @property
    def binding_feature_dim(self) -> int:
        return self.content_dim * 3 + self.relation_embed_dim + 1


@dataclass(frozen=True)
class ReadoutChannelFeatureBundle:
    """Raw pre-projection features for logging and tests."""

    trace: Tensor
    binding: Tensor
    basin: Tensor
    interference: Tensor
    lucidity: Tensor
    context: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "trace": self.trace,
            "binding": self.binding,
            "basin": self.basin,
            "interference": self.interference,
            "lucidity": self.lucidity,
            "context": self.context,
        }


@dataclass(frozen=True)
class ReadoutChannelBundle:
    """Projected readout vectors, one per cognitive-state channel."""

    trace: Tensor
    binding: Tensor
    basin: Tensor
    interference: Tensor
    lucidity: Tensor
    context: Tensor
    features: ReadoutChannelFeatureBundle | None = None

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "trace": self.trace,
            "binding": self.binding,
            "basin": self.basin,
            "interference": self.interference,
            "lucidity": self.lucidity,
            "context": self.context,
        }

    def stack(self) -> Tensor:
        return torch.stack(list(self.as_dict().values()), dim=1)


@dataclass(frozen=True)
class ReadoutChannelInspection:
    """Human-viewable readout diagnostics."""

    input_text: str
    settling_steps: int
    num_binding_edges: int
    feature_norms: dict[str, float]
    channel_norms: dict[str, float]
    top_binding_pairs: list[tuple[int, int, float]]
    top_basin_pressures: list[tuple[int, float]]
    interference_compat: float | None
    interference_conflict: float | None


def _amp_weighted_mean(content: Tensor, amps: Tensor) -> Tensor:
    weights = amps / amps.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (content * weights.unsqueeze(-1)).sum(dim=1)


def _validate_cognitive_state_input(state: CognitiveState) -> None:
    if not isinstance(state, CognitiveState):
        raise TypeError("readout channels must consume CognitiveState, not raw cue tensors.")
    if "raw_cue_shortcut" in state.meta:
        raise ValueError("readout channels reject raw cue shortcuts outside CognitiveState.")
    region = state.active_region
    if not torch.isfinite(region.trace_content).all():
        raise ValueError("active_region.trace_content contains non-finite values.")
    if not torch.isfinite(region.trace_amp).all():
        raise ValueError("active_region.trace_amp contains non-finite values.")
    if not torch.isfinite(state.basin_state.pressures).all():
        raise ValueError("basin_state.pressures contains non-finite values.")
    if not torch.isfinite(state.basin_state.vectors).all():
        raise ValueError("basin_state.vectors contains non-finite values.")


def extract_trace_features(state: CognitiveState) -> Tensor:
    """Sparse trace superposition: amplitude-weighted active trace contents."""

    region = state.active_region
    return _amp_weighted_mean(region.trace_content, region.trace_amp)


def extract_binding_features(
    state: CognitiveState,
    *,
    content_dim: int,
    num_relations: int,
    relation_embeddings: nn.Embedding,
    device: torch.device,
) -> Tensor:
    """Strength-weighted aggregate of soft bound-pair edge features."""

    binding = state.binding_state
    region = state.active_region
    dtype = region.trace_content.dtype
    batch_size = region.trace_amp.shape[0]
    feature_dim = content_dim * 3 + relation_embeddings.embedding_dim + 1
    empty = torch.zeros(batch_size, feature_dim, device=device, dtype=dtype)

    if binding is None:
        return empty

    edge_batch = gather_binding_edge_batch(
        region,
        binding,
        content_dim=content_dim,
        num_relations=num_relations,
    )
    if edge_batch.num_edges == 0:
        return empty

    hadamard = edge_batch.src_content * edge_batch.dst_content
    relation_r = relation_embeddings(edge_batch.relation_index)
    strength = edge_batch.relation_strength.unsqueeze(-1).clamp_min(0.0).to(dtype=dtype)
    pair_features = torch.cat(
        [edge_batch.src_content, edge_batch.dst_content, hadamard, relation_r, strength],
        dim=-1,
    )
    weights = edge_batch.relation_strength.clamp_min(0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (pair_features * weights.unsqueeze(-1)).sum(dim=1)


def extract_basin_features(state: CognitiveState) -> Tensor:
    """Attractor readout: settled basin pressures mixed through learnable vectors."""

    basin = state.basin_state
    return compute_basin_attractor_state(basin.pressures, basin.vectors)


def extract_interference_features(state: CognitiveState, *, device: torch.device) -> Tensor:
    """Energy-state diagnostics: compatibility, local coherence, contradiction."""

    interference = state.interference_state
    batch_size = state.active_region.trace_amp.shape[0]
    dtype = state.active_region.trace_content.dtype
    if interference is None:
        return torch.zeros(batch_size, 3, device=device, dtype=dtype)

    def _scalar(value: Tensor | None) -> Tensor:
        if value is None:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        return value

    return torch.cat(
        [
            _scalar(interference.pair_energy),
            _scalar(interference.local_energy),
            _scalar(interference.contradiction),
        ],
        dim=-1,
    )


def extract_lucidity_features(state: CognitiveState, *, device: torch.device) -> Tensor:
    lucidity = state.lucidity_state
    batch_size = state.active_region.trace_amp.shape[0]
    dtype = state.active_region.trace_content.dtype
    if lucidity is None:
        return torch.zeros(batch_size, 3, device=device, dtype=dtype)

    def _scalar(value: Tensor | None) -> Tensor:
        if value is None:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        return value

    return torch.cat(
        [
            _scalar(lucidity.score),
            _scalar(lucidity.stability),
            _scalar(lucidity.ambiguity),
        ],
        dim=-1,
    )


def extract_context_features(state: CognitiveState) -> Tensor:
    """Prompt-level context pressure from ``ContextOp``, not raw token embeddings."""

    return require_context_summary(state)


def extract_channel_features(
    state: CognitiveState,
    *,
    config: ReadoutChannelsConfig,
    relation_embeddings: nn.Embedding,
) -> ReadoutChannelFeatureBundle:
    _validate_cognitive_state_input(state)
    device = state.active_region.trace_content.device
    return ReadoutChannelFeatureBundle(
        trace=extract_trace_features(state),
        binding=extract_binding_features(
            state,
            content_dim=config.content_dim,
            num_relations=config.num_relations,
            relation_embeddings=relation_embeddings,
            device=device,
        ),
        basin=extract_basin_features(state),
        interference=extract_interference_features(state, device=device),
        lucidity=extract_lucidity_features(state, device=device),
        context=extract_context_features(state),
    )


class _ChannelProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class ReadoutChannels(nn.Module):
    """Project six cognitive-state views into a shared channel space."""

    def __init__(
        self,
        config: ReadoutChannelsConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.logger = logger
        hidden = config.resolved_hidden_dim

        self.relation_embeddings = nn.Embedding(config.num_relations, config.relation_embed_dim)
        nn.init.normal_(self.relation_embeddings.weight, mean=0.0, std=0.02)

        self.trace_projector = _ChannelProjector(config.content_dim, hidden, config.channel_dim)
        self.binding_projector = _ChannelProjector(config.binding_feature_dim, hidden, config.channel_dim)
        self.basin_projector = _ChannelProjector(config.content_dim, hidden, config.channel_dim)
        self.interference_projector = _ChannelProjector(3, hidden, config.channel_dim)
        self.lucidity_projector = _ChannelProjector(3, hidden, config.channel_dim)
        self.context_projector = _ChannelProjector(config.content_dim, hidden, config.channel_dim)

    def extract_features(self, state: CognitiveState) -> ReadoutChannelFeatureBundle:
        return extract_channel_features(
            state,
            config=self.config,
            relation_embeddings=self.relation_embeddings,
        )

    def forward(self, state: CognitiveState) -> ReadoutChannelBundle:
        features = self.extract_features(state)
        bundle = ReadoutChannelBundle(
            trace=self.trace_projector(features.trace),
            binding=self.binding_projector(features.binding),
            basin=self.basin_projector(features.basin),
            interference=self.interference_projector(features.interference),
            lucidity=self.lucidity_projector(features.lucidity),
            context=self.context_projector(features.context),
            features=features,
        )
        _log_trace(
            self.logger,
            self.config.trace,
            "readout_channels.forward",
            trace_norm=float(bundle.trace.norm(dim=-1).mean().item()),
            binding_norm=float(bundle.binding.norm(dim=-1).mean().item()),
            basin_norm=float(bundle.basin.norm(dim=-1).mean().item()),
        )
        return bundle


def attach_readout_channels(state: CognitiveState, channels: ReadoutChannelBundle) -> CognitiveState:
    state.meta["readout_channels"] = channels.as_dict()
    if channels.features is not None:
        state.meta["readout_channel_features"] = channels.features.as_dict()
    return state


def build_readout_inspection(
    state: CognitiveState,
    channels: ReadoutChannelBundle,
    *,
    input_text: str,
) -> ReadoutChannelInspection:
    binding = state.binding_state
    num_edges = 0
    top_pairs: list[tuple[int, int, float]] = []
    if binding is not None:
        num_edges = int(binding.edge_index.shape[2])
        edge_index = binding.edge_index[0].detach().cpu()
        strengths = binding.relation_strength[0].detach().cpu()
        pairs = []
        for edge in range(edge_index.shape[1]):
            strength = float(strengths[edge].item())
            if strength <= 1e-4 or not torch.isfinite(torch.tensor(strength)):
                continue
            pairs.append((int(edge_index[0, edge].item()), int(edge_index[1, edge].item()), strength))
        top_pairs = sorted(pairs, key=lambda item: item[2], reverse=True)[:6]

    pressures = state.basin_state.pressures[0].detach().cpu()
    top_basins = sorted(
        ((index, float(value.item())) for index, value in enumerate(pressures)),
        key=lambda item: item[1],
        reverse=True,
    )[:6]

    feature_norms = {}
    channel_norms = {}
    if channels.features is not None:
        for name, tensor in channels.features.as_dict().items():
            feature_norms[name] = float(tensor.norm(dim=-1).mean().item())
    for name, tensor in channels.as_dict().items():
        channel_norms[name] = float(tensor.norm(dim=-1).mean().item())

    interference = state.interference_state
    compat = conflict = None
    if interference is not None:
        if interference.pair_energy is not None:
            compat = float(interference.pair_energy[0].item())
        if interference.contradiction is not None:
            conflict = float(interference.contradiction[0].item())

    settling_steps = int(state.meta.get("settling_steps", 0))
    return ReadoutChannelInspection(
        input_text=input_text,
        settling_steps=settling_steps,
        num_binding_edges=num_edges,
        feature_norms=feature_norms,
        channel_norms=channel_norms,
        top_binding_pairs=top_pairs,
        top_basin_pressures=top_basins,
        interference_compat=compat,
        interference_conflict=conflict,
    )


def format_readout_channels_report(inspection: ReadoutChannelInspection) -> str:
    lines = [
        "\n5) READOUT CHANNELS",
        f"settling_steps={inspection.settling_steps}  binding_edges={inspection.num_binding_edges}",
        f"{'channel':<14} {'feature_norm':>12} {'projected_norm':>14}",
        "-" * 44,
    ]
    for name in READOUT_CHANNEL_NAMES:
        feature_norm = inspection.feature_norms.get(name, 0.0)
        channel_norm = inspection.channel_norms.get(name, 0.0)
        lines.append(f"{name:<14} {feature_norm:>12.4f} {channel_norm:>14.4f}")

    if inspection.top_binding_pairs:
        lines.append("")
        lines.append("top bound pairs (binding channel substrate):")
        for src, dst, strength in inspection.top_binding_pairs:
            lines.append(f"  slot_{src} -> slot_{dst}  strength={strength:.3f}")

    if inspection.top_basin_pressures:
        lines.append("")
        lines.append("top basin pressures (basin channel substrate):")
        for basin_id, pressure in inspection.top_basin_pressures:
            lines.append(f"  basin_{basin_id}  pressure={pressure:.3f}")

    if inspection.interference_compat is not None or inspection.interference_conflict is not None:
        lines.append("")
        compat = inspection.interference_compat if inspection.interference_compat is not None else 0.0
        conflict = inspection.interference_conflict if inspection.interference_conflict is not None else 0.0
        lines.append(f"interference substrate  compat={compat:.3f}  conflict={conflict:.3f}")

    return "\n".join(lines)


def inspect_readout_channels_on_state(
    state: CognitiveState,
    readout_channels: ReadoutChannels,
    *,
    input_text: str,
) -> tuple[ReadoutChannelBundle, ReadoutChannelInspection]:
    readout_channels.eval()
    with torch.no_grad():
        bundle = readout_channels(state)
    inspection = build_readout_inspection(state, bundle, input_text=input_text)
    return bundle, inspection


def run_readout_channels_on_text(
    text: str,
    *,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    num_basins: int = 32,
    settling_steps: int = 3,
    channel_dim: int = 16,
    relation_channels: int = 4,
    seed: int = 7,
    trace: bool = False,
) -> tuple[ReadoutChannelBundle, ReadoutChannelInspection]:
    cue_packet, _routing, active_region = route_text(
        text,
        num_traces=num_traces,
        top_k=top_k,
        cue_dim=cue_dim,
        seed=seed,
    )
    field_loop = FieldLoop(
        FieldLoopConfig(
            cue_dim=cue_dim,
            content_dim=cue_dim,
            active_traces=top_k,
            num_basins=num_basins,
            relation_channels=relation_channels,
            settling_steps=settling_steps,
            trace=trace,
        )
    )
    context_op = ContextOp(
        ContextOpConfig(
            cue_dim=cue_dim,
            active_traces=top_k,
            num_basins=num_basins,
        )
    )
    readout_channels = ReadoutChannels(
        ReadoutChannelsConfig(
            content_dim=cue_dim,
            channel_dim=channel_dim,
            num_relations=relation_channels,
            trace=trace,
        )
    )
    basin_state = field_loop.make_basin_state(
        active_region.trace_amp.shape[0],
        device=active_region.trace_amp.device,
    )
    field_loop.eval()
    context_op.eval()

    with torch.no_grad():
        context = context_op(cue_packet, active_region)
        field_output = field_loop(cue_packet, active_region, basin_state)
        state = build_cognitive_state_from_field_loop(
            cue_packet=cue_packet,
            active_region=active_region,
            basin_state=basin_state,
            field_output=field_output,
            context_summary=context.context_summary,
        )
        return inspect_readout_channels_on_state(state, readout_channels, input_text=text)


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect cognitive-state readout channels (E1).")
    parser.add_argument(
        "text",
        nargs="+",
        help='Text to process. You may optionally start with the word "text".',
    )
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--num-basins", type=int, default=32)
    parser.add_argument("--settling-steps", type=int, default=3)
    parser.add_argument("--channel-dim", type=int, default=16)
    parser.add_argument("--relation-channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)

    bundle, inspection = run_readout_channels_on_text(
        text,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        num_basins=args.num_basins,
        settling_steps=args.settling_steps,
        channel_dim=args.channel_dim,
        relation_channels=args.relation_channels,
        seed=args.seed,
        trace=args.trace,
    )

    _safe_print(f"Input: {text}")
    _safe_print(format_readout_channels_report(inspection))
    if args.trace and bundle.features is not None:
        _safe_print(f"trace_feature preview: {_preview_tensor(bundle.features.trace[0])}")


__all__ = [
    "READOUT_CHANNEL_NAMES",
    "ReadoutChannelBundle",
    "ReadoutChannelFeatureBundle",
    "ReadoutChannelInspection",
    "ReadoutChannels",
    "ReadoutChannelsConfig",
    "attach_readout_channels",
    "build_readout_inspection",
    "extract_basin_features",
    "extract_binding_features",
    "extract_channel_features",
    "extract_context_features",
    "extract_interference_features",
    "extract_lucidity_features",
    "extract_trace_features",
    "format_readout_channels_report",
    "inspect_readout_channels_on_state",
    "run_readout_channels_on_text",
]


if __name__ == "__main__":
    main()
