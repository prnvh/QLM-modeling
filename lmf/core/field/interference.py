"""Binding-gated interference terms for the active field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lmf.core.state.types import ActiveRegion, BasinState, BindingState, InterferenceState


@dataclass
class InterferenceLayerConfig:
    content_dim: int

    def __post_init__(self) -> None:
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")


class InterferenceLayer(nn.Module):
    """Learn compatibility/conflict energy terms gated by binding."""

    def __init__(self, config: InterferenceLayerConfig) -> None:
        super().__init__()
        self.config = config
        self.compatibility = nn.Parameter(torch.tensor(1.0))
        self.conflict = nn.Parameter(torch.tensor(0.25))

    def forward(
        self,
        active_region: ActiveRegion,
        binding_state: BindingState,
        basin_state: BasinState,
    ) -> InterferenceState:
        _ = basin_state
        content = active_region.trace_content
        if content.dim() != 3:
            raise ValueError("active_region.trace_content must have shape [batch, traces, dim].")

        normalized = F.normalize(content, dim=-1)
        pair_compat = torch.einsum("btd,btd->bt", normalized, normalized)
        local_energy = pair_compat.mean(dim=-1, keepdim=True)

        binding_mass = binding_state.relation_strength.clamp_min(0.0).mean(
            dim=-1,
            keepdim=True,
        )
        pair_energy = binding_mass * self.compatibility

        contradiction = binding_mass * self.conflict * active_region.trace_amp.std(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        return InterferenceState(
            pair_energy=pair_energy,
            local_energy=local_energy,
            contradiction=contradiction,
        )


__all__ = ["InterferenceLayer", "InterferenceLayerConfig"]
