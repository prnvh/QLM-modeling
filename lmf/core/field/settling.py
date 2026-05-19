"""Damped settling update for trace amplitudes and basin pressures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class SettlingConfig:
    eta: float = 0.1
    damping: float = 0.3
    learnable_dynamics: bool = True

    def __post_init__(self) -> None:
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must be in (0, 1].")


class Settling(nn.Module):
    """Apply bounded damped settling: proposed = sigmoid(state + eta * force)."""

    def __init__(self, config: SettlingConfig) -> None:
        super().__init__()
        self.config = config
        if config.learnable_dynamics:
            self.eta = nn.Parameter(torch.tensor(config.eta))
            raw_damping = torch.log(torch.tensor(config.damping) / (1.0 - config.damping))
            self.damping_logit = nn.Parameter(raw_damping)
        else:
            self.register_buffer("eta_fixed", torch.tensor(config.eta))
            self.register_buffer("damping_fixed", torch.tensor(config.damping))

    def forward(self, current: Tensor, force: Tensor) -> Tensor:
        if current.shape != force.shape:
            raise ValueError("settling current and force must have the same shape.")
        eta = self.eta if self.config.learnable_dynamics else self.eta_fixed
        damping = (
            torch.sigmoid(self.damping_logit)
            if self.config.learnable_dynamics
            else self.damping_fixed
        )
        proposed = torch.sigmoid(current + eta * force)
        return (1.0 - damping) * current + damping * proposed


__all__ = ["Settling", "SettlingConfig"]
