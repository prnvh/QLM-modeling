"""Commit D2: pair-derived basin support module."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from lmf.core.basin.pair_basin_support import PairDerivedBasinSupport, PairDerivedBasinSupportConfig
from lmf.core.field.binding import BindingLayer, BindingLayerConfig
from lmf.core.field.binding_forces import BindingForcesConfig, BindingForcesModule
from lmf.core.field.types import ContextPressure
from lmf.core.state.types import ActiveRegion, BasinState, BindingState


def _edge_inputs(*, batch_size: int = 1, num_edges: int = 3, dim: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(2)
    src = torch.randn(batch_size, num_edges, dim)
    dst = torch.randn(batch_size, num_edges, dim)
    relation_index = torch.tensor([[0, 1, 2]], dtype=torch.long).expand(batch_size, num_edges)
    strength = torch.tensor([[0.8, 0.5, 0.3]], dtype=torch.float32).expand(batch_size, num_edges)
    return src, dst, relation_index, strength


def test_d2_pair_feature_width_matches_plan() -> None:
    module = PairDerivedBasinSupport(
        PairDerivedBasinSupportConfig(content_dim=4, num_basins=6, num_relations=4, relation_embed_dim=8)
    )
    src, dst, relation_index, strength = _edge_inputs()
    features = module.build_pair_features(src, dst, relation_index=relation_index, binding_strength=strength)

    assert features.shape == (1, 3, 4 * 3 + 8 + 1)


def test_d2_relation_index_changes_basin_support() -> None:
    module = PairDerivedBasinSupport(
        PairDerivedBasinSupportConfig(content_dim=4, num_basins=6, num_relations=4)
    )
    src, dst, relation_index, strength = _edge_inputs()
    alt_relation = torch.zeros_like(relation_index)

    with torch.no_grad():
        first = module(src, dst, relation_index=relation_index, binding_strength=strength)
        second = module(src, dst, relation_index=alt_relation, binding_strength=strength)

    assert not torch.allclose(first, second)


def test_d2_aggregate_returns_zero_for_no_edges() -> None:
    module = PairDerivedBasinSupport(
        PairDerivedBasinSupportConfig(content_dim=4, num_basins=6, num_relations=4)
    )
    empty = torch.zeros(1, 0, 6)
    aggregated = module.aggregate_edge_support(empty, torch.zeros(1, 0))
    assert aggregated.shape == (1, 6)
    assert torch.all(aggregated == 0.0)


def test_d2_pair_basin_support_is_trainable() -> None:
    module = PairDerivedBasinSupport(
        PairDerivedBasinSupportConfig(content_dim=4, num_basins=6, num_relations=4)
    )
    src, dst, relation_index, strength = _edge_inputs()
    output = module(src, dst, relation_index=relation_index, binding_strength=strength)
    output.sum().backward()
    assert module.projection[0].weight.grad is not None


def test_d2_binding_layer_stores_relation_index() -> None:
    torch.manual_seed(5)
    layer = BindingLayer(BindingLayerConfig(content_dim=4, context_dim=4, relation_channels=4))
    region = ActiveRegion(
        trace_ids=torch.arange(4).unsqueeze(0),
        trace_content=torch.randn(1, 4, 4),
        trace_amp=torch.rand(1, 4),
        cue_drive=torch.randn(1, 4),
    )
    context = ContextPressure(
        trace_drive=torch.zeros(1, 4),
        basin_drive=torch.zeros(1, 8),
        threshold_shift=torch.zeros(1, 4),
        context_summary=torch.randn(1, 4),
    )
    basin = BasinState(pressures=torch.zeros(1, 8), vectors=torch.zeros(8, 4))
    binding = layer(region, basin, context)

    assert binding.relation_index is not None
    assert binding.relation_index.shape == binding.relation_strength.shape
    assert binding.relation_index.min() >= 0
    assert binding.relation_index.max() < 4


def test_d2_binding_forces_requires_relation_index() -> None:
    module = BindingForcesModule(BindingForcesConfig(num_basins=6, content_dim=4, num_relations=4))
    region = ActiveRegion(
        trace_ids=torch.arange(4).unsqueeze(0),
        trace_content=torch.randn(1, 4, 4),
        trace_amp=torch.rand(1, 4) + 0.1,
        cue_drive=torch.randn(1, 4),
    )
    basin = BasinState(pressures=torch.zeros(1, 6), vectors=torch.zeros(6, 4))
    binding = BindingState(
        edge_index=torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long),
        relation_strength=torch.tensor([[0.7, 0.4]]),
    )
    context = ContextPressure(
        trace_drive=torch.zeros(1, 4),
        basin_drive=torch.zeros(1, 6),
        threshold_shift=torch.zeros(1, 4),
        context_summary=torch.zeros(1, 4),
    )

    with pytest.raises(ValueError, match="relation_index"):
        module(region, basin, binding, context)


def test_d2_pair_basin_support_cli_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/core/basin/pair_basin_support.py",
            "inspect",
            "--content-dim",
            "6",
            "--num-basins",
            "8",
            "--trace",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Aggregated basin support shape: [1, 8]" in result.stdout
    assert "pair_basin_support.forward" in result.stderr
