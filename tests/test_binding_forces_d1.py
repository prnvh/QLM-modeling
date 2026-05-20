"""Commit D1: basin pressure routes through bound trace pairs, not only active traces."""

from __future__ import annotations

import pytest
import torch

from lmf.core.field.binding_forces import BindingForcesConfig, BindingForcesModule
from lmf.core.field.types import ContextPressure
from lmf.core.state.types import ActiveRegion, BasinState, BindingState


def _basin_state(*, batch_size: int = 1, num_basins: int = 6) -> BasinState:
    return BasinState(
        pressures=torch.zeros(batch_size, num_basins),
        vectors=torch.zeros(num_basins, 4),
    )


def _active_region(*, batch_size: int = 1, num_traces: int = 4, dim: int = 4) -> ActiveRegion:
    torch.manual_seed(0)
    return ActiveRegion(
        trace_ids=torch.arange(num_traces).unsqueeze(0).expand(batch_size, num_traces),
        trace_content=torch.randn(batch_size, num_traces, dim),
        trace_amp=torch.rand(batch_size, num_traces) + 0.1,
        cue_drive=torch.randn(batch_size, num_traces),
    )


def _binding_state(*, batch_size: int = 1, num_traces: int = 4, num_edges: int = 3) -> BindingState:
    edge_index = torch.tensor(
        [[[0, 1, 2], [1, 2, 3]]],
        dtype=torch.long,
    ).expand(batch_size, 2, num_edges)
    strengths = torch.tensor([[0.8, 0.5, 0.3]], dtype=torch.float32).expand(batch_size, num_edges)
    return BindingState(edge_index=edge_index, relation_strength=strengths)


def _context(*, batch_size: int = 1, dim: int = 4) -> ContextPressure:
    return ContextPressure(
        trace_drive=torch.zeros(batch_size, 4),
        basin_drive=torch.zeros(batch_size, 6),
        threshold_shift=torch.zeros(batch_size, 4),
        context_summary=torch.zeros(batch_size, dim),
    )


def test_d1_basin_force_combines_direct_and_bound_pair_paths() -> None:
    module = BindingForcesModule(
        BindingForcesConfig(
            num_basins=6,
            content_dim=4,
            direct_trace_to_basin_scale=0.2,
            bound_pair_to_basin_scale=1.0,
        )
    )
    forces = module(
        _active_region(),
        _basin_state(),
        _binding_state(),
        _context(),
    )

    assert forces.direct_basin_force is not None
    assert forces.bound_pair_basin_force is not None
    expected = 0.2 * forces.direct_basin_force + 1.0 * forces.bound_pair_basin_force
    assert torch.allclose(forces.basin_force, expected)


def test_d1_zero_binding_edges_zero_bound_pair_contribution() -> None:
    module = BindingForcesModule(BindingForcesConfig(num_basins=6, content_dim=4))
    binding = BindingState(
        edge_index=torch.zeros(1, 2, 0, dtype=torch.long),
        relation_strength=torch.zeros(1, 0),
    )
    forces = module(_active_region(), _basin_state(), binding, _context())

    assert forces.bound_pair_basin_force is not None
    assert torch.all(forces.bound_pair_basin_force == 0.0)
    assert torch.allclose(
        forces.basin_force,
        module.config.direct_trace_to_basin_scale * forces.direct_basin_force,
    )


def test_d1_disabling_bound_pair_scale_leaves_only_direct_path() -> None:
    module = BindingForcesModule(
        BindingForcesConfig(
            num_basins=6,
            content_dim=4,
            direct_trace_to_basin_scale=0.2,
            bound_pair_to_basin_scale=0.0,
        )
    )
    forces = module(_active_region(), _basin_state(), _binding_state(), _context())

    assert torch.allclose(forces.basin_force, 0.2 * forces.direct_basin_force)


def test_d1_bound_pair_path_changes_when_edge_strength_changes() -> None:
    module = BindingForcesModule(
        BindingForcesConfig(num_basins=6, content_dim=4, bound_pair_to_basin_scale=1.0)
    )
    region = _active_region()
    basin = _basin_state()
    ctx = _context()

    strong = _binding_state()
    weak = BindingState(
        edge_index=strong.edge_index,
        relation_strength=strong.relation_strength * 0.01,
    )

    strong_forces = module(region, basin, strong, ctx)
    weak_forces = module(region, basin, weak, ctx)

    assert not torch.allclose(strong_forces.bound_pair_basin_force, weak_forces.bound_pair_basin_force)


def test_d1_pair_projection_is_trainable() -> None:
    module = BindingForcesModule(BindingForcesConfig(num_basins=6, content_dim=4))
    region = _active_region()
    basin = _basin_state()
    binding = _binding_state()
    ctx = _context()

    forces = module(region, basin, binding, ctx)
    loss = forces.basin_force.sum()
    loss.backward()

    assert module.pair_basin_projection.weight.grad is not None
    assert float(module.pair_basin_projection.weight.grad.abs().sum()) > 0.0


def test_d1_config_rejects_negative_scales() -> None:
    with pytest.raises(ValueError, match="direct_trace_to_basin_scale"):
        BindingForcesConfig(num_basins=4, content_dim=4, direct_trace_to_basin_scale=-0.1)

