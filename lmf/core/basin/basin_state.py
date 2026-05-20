"""Basin state contracts, validation, and factories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lmf.core.state.types import BasinState


@dataclass(frozen=True)
class BasinStateSpec:
    """Expected tensor geometry for a runnable basin state."""

    batch_size: int
    num_basins: int
    basin_dim: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.basin_dim <= 0:
            raise ValueError("basin_dim must be positive.")


def validate_basin_state(
    basin_state: BasinState,
    *,
    spec: BasinStateSpec | None = None,
    require_finite: bool = True,
) -> None:
    """Fail loudly on malformed basin tensors before field settling runs."""

    pressures = basin_state.pressures
    vectors = basin_state.vectors

    if pressures.dim() != 2:
        raise ValueError("basin_state.pressures must have shape [batch, num_basins].")
    if not pressures.dtype.is_floating_point:
        raise ValueError("basin_state.pressures must be a floating tensor.")
    if vectors.dim() != 2:
        raise ValueError("basin_state.vectors must have shape [num_basins, basin_dim].")
    if not vectors.dtype.is_floating_point:
        raise ValueError("basin_state.vectors must be a floating tensor.")
    if pressures.shape[1] != vectors.shape[0]:
        raise ValueError("basin_state.pressures width must match basin_state.vectors rows.")

    if basin_state.basin_ids is not None:
        basin_ids = basin_state.basin_ids
        if basin_ids.dim() != 1:
            raise ValueError("basin_state.basin_ids must be one-dimensional when provided.")
        if basin_ids.shape[0] != vectors.shape[0]:
            raise ValueError("basin_state.basin_ids length must match num_basins.")

    if spec is not None:
        if pressures.shape[0] != spec.batch_size:
            raise ValueError("basin_state batch size does not match BasinStateSpec.")
        if pressures.shape[1] != spec.num_basins:
            raise ValueError("basin_state num_basins does not match BasinStateSpec.")
        if vectors.shape[1] != spec.basin_dim:
            raise ValueError("basin_state basin_dim does not match BasinStateSpec.")

    if require_finite:
        if not torch.isfinite(pressures).all():
            raise ValueError("basin_state.pressures contains non-finite values.")
        if not torch.isfinite(vectors).all():
            raise ValueError("basin_state.vectors contains non-finite values.")


def make_basin_state(
    *,
    batch_size: int,
    num_basins: int,
    basin_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    vectors: Tensor | None = None,
    pressures: Tensor | None = None,
) -> BasinState:
    """Create a validated basin state with zero initial pressures."""

    spec = BasinStateSpec(batch_size=batch_size, num_basins=num_basins, basin_dim=basin_dim)
    device = device if device is not None else torch.device("cpu")

    if pressures is None:
        pressures = torch.zeros(batch_size, num_basins, device=device, dtype=dtype)
    if vectors is None:
        vectors = torch.zeros(num_basins, basin_dim, device=device, dtype=dtype)

    basin_state = BasinState(
        pressures=pressures,
        vectors=vectors,
        basin_ids=torch.arange(num_basins, device=device, dtype=torch.long),
    )
    validate_basin_state(basin_state, spec=spec, require_finite=True)
    return basin_state


__all__ = ["BasinStateSpec", "make_basin_state", "validate_basin_state"]
