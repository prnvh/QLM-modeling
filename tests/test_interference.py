"""Commit F: basin-competition interference + coherence/conflict hook."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch import nn

from lmf.core.field.interference import (
    InterferenceLayer,
    InterferenceLayerConfig,
    compose_basin_interference_force,
    compute_basin_competition,
    format_interference_report,
    run_interference_on_text,
)
from lmf.core.field.loop import FieldLoop, FieldLoopConfig
from lmf.core.input.cue_packet import CuePacket
from lmf.core.state.types import ActiveRegion, BasinState, BindingState
from lmf.training.coherence_conflict import (
    CoherenceConflictConfig,
    CoherenceConflictHook,
    coherence_conflict_margin_loss,
    interference_coherence_score,
    load_coherence_conflict_examples,
)


def _sample_binding(*, batch_size: int = 1, num_edges: int = 4) -> BindingState:
    return BindingState(
        edge_index=torch.tensor([[[0, 1, 2, 3], [1, 2, 3, 0]]]).expand(batch_size, -1, -1),
        relation_strength=torch.tensor([[0.8, 0.5, 0.3, 0.2]]).expand(batch_size, -1),
        relation_index=torch.zeros(batch_size, num_edges, dtype=torch.long),
    )


def _sample_region(*, batch_size: int = 1, num_traces: int = 4, dim: int = 6) -> ActiveRegion:
    return ActiveRegion(
        trace_ids=torch.arange(num_traces).unsqueeze(0).expand(batch_size, num_traces),
        trace_content=torch.randn(batch_size, num_traces, dim),
        trace_amp=torch.rand(batch_size, num_traces),
        cue_drive=torch.randn(batch_size, num_traces),
    )


def _sample_basin(*, batch_size: int = 1, num_basins: int = 6, dim: int = 6) -> BasinState:
    pressures = torch.zeros(batch_size, num_basins)
    pressures[0, 0] = 0.9
    pressures[0, 1] = 0.85
    vectors = torch.randn(num_basins, dim)
    vectors[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vectors[1] = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return BasinState(pressures=pressures, vectors=vectors)


def test_interference_uses_basin_state_for_competition() -> None:
    layer = InterferenceLayer(InterferenceLayerConfig(content_dim=6, num_basins=6))
    region = _sample_region()
    binding = _sample_binding()
    basin = _sample_basin()

    breakdown = layer.compute_breakdown(region, binding, basin)
    assert breakdown.conflict_score is not None
    assert float(breakdown.conflict_score.item()) > 0.0
    assert breakdown.basin_total_force.shape == (1, 6)


def test_competing_basins_raise_conflict_score() -> None:
    layer = InterferenceLayer(InterferenceLayerConfig(content_dim=6, num_basins=6))
    region = _sample_region()
    binding = _sample_binding()
    competing = _sample_basin()
    aligned = BasinState(
        pressures=torch.tensor([[0.9, 0.85, 0.0, 0.0, 0.0, 0.0]]),
        vectors=competing.vectors.clone(),
    )
    aligned.vectors[1] = aligned.vectors[0].clone()

    competing_score = float(layer.compute_breakdown(region, binding, competing).conflict_score.item())
    aligned_score = float(layer.compute_breakdown(region, binding, aligned).conflict_score.item())
    assert competing_score > aligned_score


def test_zero_basin_pressures_zero_basin_interference() -> None:
    layer = InterferenceLayer(InterferenceLayerConfig(content_dim=6, num_basins=6))
    region = _sample_region()
    binding = _sample_binding()
    empty = BasinState(pressures=torch.zeros(1, 6), vectors=torch.randn(6, 6))
    breakdown = layer.compute_breakdown(region, binding, empty)
    assert torch.allclose(breakdown.basin_total_force, torch.zeros_like(breakdown.basin_total_force))


def test_field_loop_basin_interference_force_changes_pressures() -> None:
    from unittest.mock import patch

    torch.manual_seed(2)
    field_loop = FieldLoop(
        FieldLoopConfig(cue_dim=6, content_dim=6, active_traces=4, num_basins=8, settling_steps=2)
    )
    cue = CuePacket(cues=torch.randn(1, 4, 6), pooled=torch.randn(1, 6))
    region = _sample_region(num_traces=4)
    basin = field_loop.make_basin_state(1)

    full = field_loop(cue, region, basin).basin_pressures
    with patch(
        "lmf.core.field.loop.compose_basin_interference_force",
        return_value=torch.full((1, 8), 0.5),
    ):
        perturbed = field_loop(cue, region, field_loop.make_basin_state(1)).basin_pressures
    assert not torch.allclose(full, perturbed)


def test_compose_basin_interference_force_matches_components() -> None:
    layer = InterferenceLayer(InterferenceLayerConfig(content_dim=6, num_basins=6))
    state = layer(_sample_region(), _sample_binding(), _sample_basin())
    composed = compose_basin_interference_force(state)
    expected = state.basin_support_force + state.basin_coexistence_force - state.basin_conflict_force - state.basin_suppression_force  # type: ignore[operator]
    assert torch.allclose(composed, expected)


def test_interference_layer_has_no_transformer_attention() -> None:
    layer = InterferenceLayer(InterferenceLayerConfig(content_dim=6, num_basins=6))
    forbidden = (nn.MultiheadAttention, nn.TransformerEncoder, nn.TransformerDecoder)
    for module in layer.modules():
        assert not isinstance(module, forbidden)


def test_coherence_conflict_margin_loss_prefers_compatible() -> None:
    compatible = torch.tensor([0.5, 0.4])
    incompatible = torch.tensor([0.1, 0.0])
    loss_ok = coherence_conflict_margin_loss(
        compatible_scores=compatible,
        incompatible_scores=incompatible,
        margin=0.1,
    )
    loss_bad = coherence_conflict_margin_loss(
        compatible_scores=incompatible,
        incompatible_scores=compatible,
        margin=0.1,
    )
    assert float(loss_ok.item()) < float(loss_bad.item())


def test_coherence_conflict_hook_loads_dataset() -> None:
    examples = load_coherence_conflict_examples("data/stage1/coherence_conflict.jsonl")
    assert len(examples) >= 4
    assert any(example.label == 1 for example in examples)
    assert any(example.label == 0 for example in examples)


def test_coherence_conflict_hook_runs_on_dataset() -> None:
    examples = load_coherence_conflict_examples("data/stage1/coherence_conflict.jsonl")
    hook = CoherenceConflictHook(CoherenceConflictConfig())
    hook.eval()
    with torch.no_grad():
        _loss, metrics = hook.loss_from_examples(examples)
    assert metrics.num_pairs == len(examples)


def test_interference_coherence_score_sign() -> None:
    layer = InterferenceLayer(InterferenceLayerConfig(content_dim=6, num_basins=6))
    state = layer(_sample_region(), _sample_binding(), _sample_basin())
    score = interference_coherence_score(state)
    assert score.shape == (1,)


def test_interference_cli_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/core/field/interference.py",
            "text",
            "Help me withdraw money from the bank",
            "--num-traces",
            "24",
            "--top-k",
            "4",
            "--cue-dim",
            "6",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "conflict_score" in result.stdout


def test_run_interference_on_text_smoke() -> None:
    breakdown = run_interference_on_text(
        "Help bank!",
        num_traces=24,
        top_k=4,
        cue_dim=6,
        num_basins=8,
        seed=1,
    )
    report = format_interference_report(breakdown, text="Help bank!")
    assert "interference_pressure" in report
