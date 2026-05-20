"""Adversarial contract tests for the basin stack."""

from __future__ import annotations

import io
import subprocess
import sys

import pytest
import torch

from lmf.core.basin.basin_bank import BasinBank, BasinBankConfig
from lmf.core.basin.basin_forces import BasinForceComposer, BasinForceComposerConfig
from lmf.core.basin.basin_state import BasinStateSpec, make_basin_state, validate_basin_state
from lmf.core.basin.binding_edges import gather_binding_edge_batch, validate_binding_state
from lmf.core.basin.pair_basin_support import PairDerivedBasinSupport, PairDerivedBasinSupportConfig
from lmf.core.field.binding_forces import BindingForcesConfig, BindingForcesModule
from lmf.core.field.loop import FieldLoop, FieldLoopConfig, make_placeholder_basin_state
from lmf.core.field.types import ContextPressure
from lmf.core.state.types import ActiveRegion, BasinState, BindingState


def _region(*, batch: int = 1, traces: int = 4, dim: int = 4) -> ActiveRegion:
    return ActiveRegion(
        trace_ids=torch.arange(traces).unsqueeze(0).expand(batch, traces),
        trace_content=torch.randn(batch, traces, dim),
        trace_amp=torch.rand(batch, traces) + 0.1,
        cue_drive=torch.randn(batch, traces),
    )


def _binding(*, batch: int = 1, edges: int = 2) -> BindingState:
    return BindingState(
        edge_index=torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long).expand(batch, 2, edges),
        relation_strength=torch.tensor([[0.7, 0.4]], dtype=torch.float32).expand(batch, edges),
        relation_index=torch.tensor([[0, 2]], dtype=torch.long).expand(batch, edges),
    )


def test_validate_basin_state_rejects_shape_mismatch() -> None:
    state = BasinState(
        pressures=torch.zeros(2, 5),
        vectors=torch.zeros(4, 6),
    )
    with pytest.raises(ValueError, match="pressures width"):
        validate_basin_state(state)


def test_validate_basin_state_rejects_non_finite() -> None:
    state = make_basin_state(batch_size=1, num_basins=4, basin_dim=3)
    state.pressures[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_basin_state(state)


def test_validate_binding_state_rejects_out_of_range_edge() -> None:
    binding = BindingState(
        edge_index=torch.tensor([[[0], [9]]], dtype=torch.long),
        relation_strength=torch.tensor([[0.5]]),
        relation_index=torch.tensor([[0]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="num_traces"):
        validate_binding_state(binding, batch_size=1, num_traces=4, num_relations=4)


def test_validate_binding_state_rejects_bad_relation_index() -> None:
    binding = BindingState(
        edge_index=torch.tensor([[[0], [1]]], dtype=torch.long),
        relation_strength=torch.tensor([[0.5]]),
        relation_index=torch.tensor([[9]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="relation_index"):
        validate_binding_state(binding, batch_size=1, num_traces=4, num_relations=4)


def test_pair_basin_support_rejects_out_of_range_relation_index() -> None:
    module = PairDerivedBasinSupport(
        PairDerivedBasinSupportConfig(content_dim=4, num_basins=6, num_relations=3)
    )
    src = torch.randn(1, 1, 4)
    dst = torch.randn(1, 1, 4)
    with pytest.raises(ValueError, match="relation_index"):
        module(src, dst, relation_index=torch.tensor([[5]]), binding_strength=torch.tensor([[0.5]]))


def test_basin_force_composer_respects_trace_mask() -> None:
    composer = BasinForceComposer(BasinForceComposerConfig(num_basins=4, content_dim=4, num_relations=3))
    region = _region()
    region.mask = torch.tensor([[True, False, False, False]])
    basin = make_basin_state(batch_size=1, num_basins=4, basin_dim=4)
    binding = BindingState(
        edge_index=torch.zeros(1, 2, 0, dtype=torch.long),
        relation_strength=torch.zeros(1, 0),
    )

    breakdown = composer(region, basin, binding)
    assert breakdown.direct.shape == (1, 4)
    assert torch.isfinite(breakdown.total).all()


def test_field_loop_rejects_mismatched_basin_vectors() -> None:
    from lmf.core.input.cue_packet import CuePacket

    field_loop = FieldLoop(FieldLoopConfig(cue_dim=4, content_dim=4, active_traces=3, num_basins=5))
    wrong = make_placeholder_basin_state(
        batch_size=1,
        num_basins=5,
        basin_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cue_packet = CuePacket(
        cues=torch.randn(1, 3, 4),
        mask=torch.ones(1, 3, dtype=torch.bool),
        pooled=torch.randn(1, 4),
    )
    with pytest.raises(ValueError, match="basin_bank.vectors"):
        field_loop(cue_packet, _region(batch=1, traces=3, dim=4), wrong)


def test_field_loop_basin_bank_is_checkpointed() -> None:
    field_loop = FieldLoop(FieldLoopConfig(cue_dim=4, content_dim=4, active_traces=3, num_basins=5))
    assert any(name.startswith("basin_bank.") for name, _ in field_loop.named_parameters())

    buffer = io.BytesIO()
    torch.save(field_loop.state_dict(), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    reloaded = FieldLoop(FieldLoopConfig(cue_dim=4, content_dim=4, active_traces=3, num_basins=5))
    reloaded.load_state_dict(restored)
    assert torch.equal(field_loop.basin_bank.vectors, reloaded.basin_bank.vectors)


def test_gather_binding_edge_batch_matches_manual_gather() -> None:
    region = _region()
    binding = _binding()
    batch = gather_binding_edge_batch(region, binding, content_dim=4, num_relations=4)
    manual_src = region.trace_content[0, binding.edge_index[0, 0, 0]]
    assert torch.allclose(batch.src_content[0, 0], manual_src)


def test_basin_bank_cli_runs() -> None:
    result = subprocess.run(
        [sys.executable, "lmf/core/basin/basin_bank.py", "inspect", "--num-basins", "6", "--trace"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Basin bank vectors shape: [6," in result.stdout
    assert "basin_bank.batch_state" in result.stderr


def test_binding_forces_gradients_reach_basin_stack() -> None:
    module = BindingForcesModule(BindingForcesConfig(num_basins=5, content_dim=4, num_relations=3))
    basin = make_basin_state(batch_size=1, num_basins=5, basin_dim=4)
    forces = module(
        _region(),
        basin,
        _binding(),
        ContextPressure(
            trace_drive=torch.zeros(1, 4),
            basin_drive=torch.zeros(1, 5),
            threshold_shift=torch.zeros(1, 4),
            context_summary=torch.zeros(1, 4),
        ),
    )
    forces.basin_force.sum().backward()
    assert module.basin_composer.pair_basin_support.projection[0].weight.grad is not None
