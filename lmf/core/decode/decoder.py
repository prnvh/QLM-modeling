"""Channelized cognitive-state decoder (Commit E2/E3 fusion on E1 readout channels)."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.decode.readout_channels import (  # noqa: E402
    ReadoutChannelBundle,
    ReadoutChannels,
    ReadoutChannelsConfig,
    attach_readout_channels,
    format_readout_channels_report,
    run_readout_channels_on_text,
)
from lmf.core.dmf.trace_router import route_text  # noqa: E402
from lmf.core.field.context_op import ContextOp, ContextOpConfig  # noqa: E402
from lmf.core.field.loop import FieldLoop, FieldLoopConfig  # noqa: E402
from lmf.core.state.cognitive_state import build_cognitive_state_from_field_loop  # noqa: E402
from lmf.core.state.types import CognitiveState  # noqa: E402

from lmf.core.decode.readout_channels import READOUT_CHANNEL_NAMES  # noqa: E402

LOGGER = logging.getLogger(__name__)

CHANNEL_NAMES = READOUT_CHANNEL_NAMES


@dataclass
class ReadoutDropoutConfig:
    trace: float = 0.05
    binding: float = 0.10
    basin: float = 0.10
    interference: float = 0.10
    lucidity: float = 0.0
    context: float = 0.0

    def __post_init__(self) -> None:
        for name in CHANNEL_NAMES:
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"readout_dropout.{name} must be in [0, 1). Got {value}.")


@dataclass
class DecoderConfig:
    content_dim: int
    readout_dim: int
    num_relations: int = 4
    trace_channel_scale: float = 0.7
    binding_channel_scale: float = 1.0
    basin_channel_scale: float = 1.3
    interference_channel_scale: float = 1.0
    lucidity_channel_scale: float = 1.0
    context_channel_scale: float = 1.0
    readout_dropout: ReadoutDropoutConfig = field(default_factory=ReadoutDropoutConfig)
    hidden_dim: int | None = None
    no_direct_cue_shortcut: bool = True

    def __post_init__(self) -> None:
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.readout_dim <= 0:
            raise ValueError("readout_dim must be positive.")
        for name in CHANNEL_NAMES:
            scale = getattr(self, f"{name}_channel_scale")
            if scale < 0.0:
                raise ValueError(f"{name}_channel_scale must be non-negative.")

    def readout_channels_config(self) -> ReadoutChannelsConfig:
        return ReadoutChannelsConfig(
            content_dim=self.content_dim,
            channel_dim=self.readout_dim,
            num_relations=self.num_relations,
            hidden_dim=self.hidden_dim,
        )


@dataclass(frozen=True)
class ChannelAblation:
    """Same-checkpoint channel mask for eval ablations."""

    trace: bool = True
    binding: bool = True
    basin: bool = True
    interference: bool = True
    lucidity: bool = True
    context: bool = True

    def enabled_channels(self) -> tuple[str, ...]:
        return tuple(name for name in CHANNEL_NAMES if getattr(self, name))

    def with_only(self, *channels: str) -> ChannelAblation:
        allowed = set(channels)
        unknown = allowed.difference(CHANNEL_NAMES)
        if unknown:
            raise ValueError(f"Unknown channel(s): {sorted(unknown)}")
        return ChannelAblation(
            trace="trace" in allowed,
            binding="binding" in allowed,
            basin="basin" in allowed,
            interference="interference" in allowed,
            lucidity="lucidity" in allowed,
            context="context" in allowed,
        )


FULL_STATE_ABLATION = ChannelAblation()
TRACE_ONLY_ABLATION = ChannelAblation().with_only("trace")
ZERO_BINDING_ABLATION = replace(FULL_STATE_ABLATION, binding=False)
ZERO_BASINS_ABLATION = replace(FULL_STATE_ABLATION, basin=False)
ZERO_INTERFERENCE_ABLATION = replace(FULL_STATE_ABLATION, interference=False)
ZERO_LUCIDITY_ABLATION = replace(FULL_STATE_ABLATION, lucidity=False)

STANDARD_DECODER_ABLATIONS: Mapping[str, ChannelAblation] = {
    "full": FULL_STATE_ABLATION,
    "trace_only": TRACE_ONLY_ABLATION,
    "zero_binding": ZERO_BINDING_ABLATION,
    "zero_basins": ZERO_BASINS_ABLATION,
    "zero_interference": ZERO_INTERFERENCE_ABLATION,
    "zero_lucidity": ZERO_LUCIDITY_ABLATION,
}


@dataclass(frozen=True)
class DecoderOutput:
    readout: Tensor
    channel_vectors: Mapping[str, Tensor]
    readout_channels: ReadoutChannelBundle | None = None


class CognitiveStateDecoder(nn.Module):
    """Fuse E1 readout channels with E2 scales/dropout into one readout vector."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.readout_channels = ReadoutChannels(config.readout_channels_config())

        dropout = config.readout_dropout
        self.trace_dropout = nn.Dropout(dropout.trace)
        self.binding_dropout = nn.Dropout(dropout.binding)
        self.basin_dropout = nn.Dropout(dropout.basin)
        self.interference_dropout = nn.Dropout(dropout.interference)
        self.lucidity_dropout = nn.Dropout(dropout.lucidity)
        self.context_dropout = nn.Dropout(dropout.context)

    def project_channels(
        self,
        state: CognitiveState,
        *,
        ablation: ChannelAblation | None = None,
    ) -> tuple[ReadoutChannelBundle, dict[str, Tensor]]:
        if self.config.no_direct_cue_shortcut and "raw_cue_shortcut" in state.meta:
            raise ValueError("decoder rejects raw cue shortcuts outside CognitiveState.")

        ablation = ablation or FULL_STATE_ABLATION
        channels = self.readout_channels(state)
        channel_map = channels.as_dict()

        dropouts = {
            "trace": self.trace_dropout,
            "binding": self.binding_dropout,
            "basin": self.basin_dropout,
            "interference": self.interference_dropout,
            "lucidity": self.lucidity_dropout,
            "context": self.context_dropout,
        }
        scales = {
            "trace": self.config.trace_channel_scale,
            "binding": self.config.binding_channel_scale,
            "basin": self.config.basin_channel_scale,
            "interference": self.config.interference_channel_scale,
            "lucidity": self.config.lucidity_channel_scale,
            "context": self.config.context_channel_scale,
        }

        fused: dict[str, Tensor] = {}
        for name in CHANNEL_NAMES:
            projected = dropouts[name](channel_map[name])
            if getattr(ablation, name):
                projected = projected * scales[name]
            else:
                projected = torch.zeros_like(projected)
            fused[name] = projected

        return channels, fused

    def forward(
        self,
        state: CognitiveState,
        *,
        ablation: ChannelAblation | None = None,
        attach_channels: bool = False,
    ) -> DecoderOutput:
        channels, fused = self.project_channels(state, ablation=ablation)
        if attach_channels:
            attach_readout_channels(state, channels)

        batch_size = channels.trace.shape[0]
        device = channels.trace.device
        readout = torch.zeros(batch_size, self.config.readout_dim, device=device, dtype=channels.trace.dtype)
        for name in CHANNEL_NAMES:
            readout = readout + fused[name]

        readout = F.normalize(readout, dim=-1, eps=1e-6)
        return DecoderOutput(
            readout=readout,
            channel_vectors=fused,
            readout_channels=channels,
        )


def decoder_config_from_mapping(raw: Mapping[str, object], *, content_dim: int, readout_dim: int) -> DecoderConfig:
    """Build ``DecoderConfig`` from a yaml-style mapping."""

    dropout_raw = raw.get("readout_dropout", {})
    if not isinstance(dropout_raw, Mapping):
        raise TypeError("readout_dropout must be a mapping.")
    dropout = ReadoutDropoutConfig(
        trace=float(dropout_raw.get("trace", 0.05)),
        binding=float(dropout_raw.get("binding", 0.10)),
        basin=float(dropout_raw.get("basin", 0.10)),
        interference=float(dropout_raw.get("interference", 0.10)),
        lucidity=float(dropout_raw.get("lucidity", 0.0)),
        context=float(dropout_raw.get("context", 0.0)),
    )
    return DecoderConfig(
        content_dim=content_dim,
        readout_dim=readout_dim,
        trace_channel_scale=float(raw.get("trace_channel_scale", 0.7)),
        binding_channel_scale=float(raw.get("binding_channel_scale", 1.0)),
        basin_channel_scale=float(raw.get("basin_channel_scale", 1.3)),
        interference_channel_scale=float(raw.get("interference_channel_scale", 1.0)),
        lucidity_channel_scale=float(raw.get("lucidity_channel_scale", 1.0)),
        context_channel_scale=float(raw.get("context_channel_scale", 1.0)),
        readout_dropout=dropout,
        hidden_dim=int(raw["hidden_dim"]) if "hidden_dim" in raw else None,
        no_direct_cue_shortcut=bool(raw.get("no_direct_cue_shortcut", True)),
    )


def run_decoder_on_text(
    text: str,
    *,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    num_basins: int = 32,
    settling_steps: int = 3,
    readout_dim: int = 16,
    seed: int = 7,
    ablation: ChannelAblation | None = None,
) -> DecoderOutput:
    """Route text through the field loop and decode the resulting cognitive state."""

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
            settling_steps=settling_steps,
        )
    )
    context_op = ContextOp(
        ContextOpConfig(
            cue_dim=cue_dim,
            active_traces=top_k,
            num_basins=num_basins,
        )
    )
    basin_state = field_loop.make_basin_state(
        active_region.trace_amp.shape[0],
        device=active_region.trace_amp.device,
    )
    field_loop.eval()
    context_op.eval()
    decoder = CognitiveStateDecoder(
        DecoderConfig(content_dim=cue_dim, readout_dim=readout_dim)
    )
    decoder.eval()

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
        return decoder(state, ablation=ablation)


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode cognitive state from text.")
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
    parser.add_argument("--readout-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--ablation",
        choices=sorted(STANDARD_DECODER_ABLATIONS),
        default="full",
    )
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)

    output = run_decoder_on_text(
        text,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        num_basins=args.num_basins,
        settling_steps=args.settling_steps,
        readout_dim=args.readout_dim,
        seed=args.seed,
        ablation=STANDARD_DECODER_ABLATIONS[args.ablation],
    )

    _safe_print(f"Input: {text}")
    _safe_print(f"Ablation: {args.ablation}")
    _safe_print(f"Readout shape: {list(output.readout.shape)}")
    _safe_print(f"Readout preview: {output.readout[0, :6].tolist()}")
    for name in CHANNEL_NAMES:
        vector = output.channel_vectors[name]
        _safe_print(f"{name}_channel_norm: {float(vector.norm(dim=-1).mean().item()):.4f}")


__all__ = [
    "CHANNEL_NAMES",
    "ChannelAblation",
    "CognitiveStateDecoder",
    "DecoderConfig",
    "DecoderOutput",
    "FULL_STATE_ABLATION",
    "ReadoutDropoutConfig",
    "STANDARD_DECODER_ABLATIONS",
    "TRACE_ONLY_ABLATION",
    "ZERO_BINDING_ABLATION",
    "ZERO_BASINS_ABLATION",
    "ZERO_INTERFERENCE_ABLATION",
    "ZERO_LUCIDITY_ABLATION",
    "decoder_config_from_mapping",
    "run_decoder_on_text",
]

# Re-export E1 symbols for convenience.
__all__ += [
    "ReadoutChannelBundle",
    "ReadoutChannels",
    "ReadoutChannelsConfig",
    "attach_readout_channels",
    "run_readout_channels_on_text",
]


if __name__ == "__main__":
    main()
