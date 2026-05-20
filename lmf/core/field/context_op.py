"""Context operator: cue summary applies pressure to the active field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lmf.core.field.types import ContextPressure
from lmf.core.input.cue_packet import CuePacket
from lmf.core.state.types import ActiveRegion


@dataclass
class ContextOpConfig:
    cue_dim: int
    active_traces: int
    num_basins: int
    hidden_dim: int | None = None

    def __post_init__(self) -> None:
        if self.cue_dim <= 0:
            raise ValueError("cue_dim must be positive.")
        if self.active_traces <= 0:
            raise ValueError("active_traces must be positive.")
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")


class ContextOp(nn.Module):
    """Map pooled cues to field pressures without token-token attention."""

    def __init__(self, config: ContextOpConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim if config.hidden_dim is not None else max(config.cue_dim, 16)

        out_dim = config.active_traces + config.num_basins + config.active_traces
        self.projection = nn.Sequential(
            nn.Linear(config.cue_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, cue_packet: CuePacket, active_region: ActiveRegion) -> ContextPressure:
        summary = self._resolve_summary(cue_packet)
        projected = self.projection(summary)

        k = active_region.trace_amp.shape[-1]
        num_basins = self.config.num_basins
        trace_drive = projected[:, :k]
        basin_drive = projected[:, k : k + num_basins]
        threshold_shift = projected[:, k + num_basins : k + num_basins + k]

        if trace_drive.shape != active_region.trace_amp.shape:
            raise ValueError("context trace_drive must match active_region.trace_amp shape.")

        return ContextPressure(
            trace_drive=trace_drive,
            basin_drive=basin_drive,
            threshold_shift=threshold_shift,
            context_summary=summary,
        )

    def _resolve_summary(self, cue_packet: CuePacket) -> Tensor:
        if cue_packet.pooled is not None:
            pooled = cue_packet.pooled
            if pooled.dim() == 1:
                pooled = pooled.unsqueeze(0)
            return pooled

        cues = cue_packet.cues
        if cues.dim() == 2:
            cues = cues.unsqueeze(0)
        if cue_packet.mask is None:
            return cues.mean(dim=1)
        mask = cue_packet.mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=cues.dtype)
        return (cues * mask.unsqueeze(-1).to(dtype=cues.dtype)).sum(dim=1) / counts


__all__ = ["ContextOp", "ContextOpConfig"]
