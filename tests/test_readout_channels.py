"""Commit E1: cognitive-state readout channels."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from lmf.core.decode.readout_channels import (
    READOUT_CHANNEL_NAMES,
    ReadoutChannelBundle,
    ReadoutChannels,
    ReadoutChannelsConfig,
    attach_readout_channels,
    format_readout_channels_report,
    run_readout_channels_on_text,
)
from lmf.core.state.cognitive_state import build_cognitive_state, require_context_summary
from lmf.core.state.types import (
    ActiveRegion,
    BasinState,
    BindingState,
    CognitiveState,
    CuePacket,
    InterferenceState,
    LucidityState,
)


def _sample_state(*, batch_size: int = 2, num_traces: int = 5, dim: int = 6) -> CognitiveState:
    return build_cognitive_state(
        cue_packet=CuePacket(
            cues=torch.randn(batch_size, 4, dim),
            mask=torch.ones(batch_size, 4, dtype=torch.bool),
            pooled=torch.randn(batch_size, dim),
        ),
        active_region=ActiveRegion(
            trace_ids=torch.arange(num_traces).unsqueeze(0).expand(batch_size, num_traces),
            trace_content=torch.randn(batch_size, num_traces, dim),
            trace_amp=torch.rand(batch_size, num_traces),
            cue_drive=torch.randn(batch_size, num_traces),
            mask=torch.ones(batch_size, num_traces, dtype=torch.bool),
        ),
        basin_state=BasinState(
            pressures=torch.rand(batch_size, 8),
            vectors=torch.randn(8, dim),
        ),
        binding_state=BindingState(
            edge_index=torch.tensor([[[0, 1, 2], [1, 2, 0]]]).expand(batch_size, -1, -1),
            relation_strength=torch.tensor([[0.9, 0.4, 0.2]]).expand(batch_size, -1),
            relation_index=torch.tensor([[0, 1, 2]]).expand(batch_size, -1),
            centrality=torch.softmax(torch.rand(batch_size, num_traces), dim=-1),
        ),
        interference_state=InterferenceState(
            pair_energy=torch.rand(batch_size, 1),
            local_energy=torch.rand(batch_size, 1),
            contradiction=torch.rand(batch_size, 1),
        ),
        lucidity_state=LucidityState(
            score=torch.rand(batch_size, 1),
            stability=torch.rand(batch_size, 1),
            ambiguity=torch.rand(batch_size, 1),
        ),
        context_summary=torch.randn(batch_size, dim),
    )


def test_readout_channel_names_match_commit_plan() -> None:
    assert READOUT_CHANNEL_NAMES == (
        "trace",
        "binding",
        "basin",
        "interference",
        "lucidity",
        "context",
    )


def test_extract_channel_features_returns_all_views() -> None:
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    state = _sample_state(batch_size=1)
    features = module.extract_features(state)
    assert set(features.as_dict()) == set(READOUT_CHANNEL_NAMES)
    assert features.trace.shape == (1, 6)
    assert features.binding.shape == (1, module.config.binding_feature_dim)
    assert features.interference.shape == (1, 3)
    assert features.context.shape == (1, 6)


def test_readout_channels_project_to_shared_dim() -> None:
    torch.manual_seed(0)
    state = _sample_state()
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    bundle = module(state)

    assert isinstance(bundle, ReadoutChannelBundle)
    assert bundle.trace.shape == (2, 8)
    assert bundle.stack().shape == (2, 6, 8)
    assert bundle.features is not None
    for name in READOUT_CHANNEL_NAMES:
        assert torch.isfinite(getattr(bundle, name)).all()


def test_binding_features_use_bound_pair_edges_not_centrality_only() -> None:
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    state = _sample_state(batch_size=1)
    baseline = module.extract_features(state).binding

    strong_binding = BindingState(
        edge_index=state.binding_state.edge_index,  # type: ignore[union-attr]
        relation_strength=torch.tensor([[0.99, 0.01, 0.01]]),
        relation_index=state.binding_state.relation_index,  # type: ignore[union-attr]
    )
    swapped = build_cognitive_state(
        cue_packet=state.cue_packet,
        active_region=state.active_region,
        basin_state=state.basin_state,
        binding_state=strong_binding,
        interference_state=state.interference_state,
        lucidity_state=state.lucidity_state,
        context_summary=require_context_summary(state),
    )
    changed = module.extract_features(swapped).binding
    assert not torch.allclose(baseline, changed)


def test_binding_features_zero_when_no_edges() -> None:
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    state = _sample_state(batch_size=1)
    empty_binding = BindingState(
        edge_index=torch.zeros(1, 2, 0, dtype=torch.long),
        relation_strength=torch.zeros(1, 0),
        relation_index=torch.zeros(1, 0, dtype=torch.long),
    )
    empty_state = build_cognitive_state(
        cue_packet=state.cue_packet,
        active_region=state.active_region,
        basin_state=state.basin_state,
        binding_state=empty_binding,
        interference_state=state.interference_state,
        context_summary=require_context_summary(state),
    )
    features = module.extract_features(empty_state)
    assert torch.allclose(features.binding, torch.zeros_like(features.binding))


def test_trace_features_ignore_cue_packet_when_active_region_fixed() -> None:
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    state = _sample_state(batch_size=1)
    baseline = module.extract_features(state).trace
    state.cue_packet = CuePacket(cues=torch.zeros(1, 4, 6), pooled=torch.zeros(1, 6))
    unchanged = module.extract_features(state).trace
    assert torch.allclose(baseline, unchanged)


def test_channels_are_not_identical_projections() -> None:
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    bundle = module(_sample_state(batch_size=1))
    trace = F.normalize(bundle.trace, dim=-1)
    binding = F.normalize(bundle.binding, dim=-1)
    basin = F.normalize(bundle.basin, dim=-1)
    assert float((1.0 - (trace * binding).sum()).item()) > 0.05
    assert float((1.0 - (trace * basin).sum()).item()) > 0.05


def test_readout_channels_reject_raw_cue_shortcut() -> None:
    state = _sample_state(batch_size=1)
    state.meta["raw_cue_shortcut"] = True
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    with pytest.raises(ValueError, match="raw cue shortcuts"):
        module(state)


def test_readout_channels_require_context_summary() -> None:
    state = _sample_state(batch_size=1)
    state.meta.pop("context_summary")
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    with pytest.raises(ValueError, match="context_summary"):
        module(state)


def test_readout_channels_reject_invalid_binding_relation_index() -> None:
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8, num_relations=4))
    state = _sample_state(batch_size=1)
    bad_binding = BindingState(
        edge_index=state.binding_state.edge_index,  # type: ignore[union-attr]
        relation_strength=state.binding_state.relation_strength,  # type: ignore[union-attr]
        relation_index=torch.tensor([[99]]),
    )
    bad_state = build_cognitive_state(
        cue_packet=state.cue_packet,
        active_region=state.active_region,
        basin_state=state.basin_state,
        binding_state=bad_binding,
        context_summary=require_context_summary(state),
    )
    with pytest.raises(ValueError, match="relation_index"):
        module.extract_features(bad_state)


def test_readout_channels_do_not_use_transformer_layers() -> None:
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    forbidden = (nn.MultiheadAttention, nn.TransformerEncoder, nn.TransformerDecoder)
    for submodule in module.modules():
        assert not isinstance(submodule, forbidden)


def test_readout_channels_support_gradients() -> None:
    state = _sample_state(batch_size=1)
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    bundle = module(state)
    sum(getattr(bundle, name).sum() for name in READOUT_CHANNEL_NAMES).backward()
    assert module.basin_projector.net[0].weight.grad is not None
    assert module.relation_embeddings.weight.grad is not None


def test_attach_readout_channels_stores_on_meta() -> None:
    state = _sample_state(batch_size=1)
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    bundle = module(state)
    attach_readout_channels(state, bundle)
    stored = state.meta["readout_channels"]
    assert set(stored) == set(READOUT_CHANNEL_NAMES)
    assert "readout_channel_features" in state.meta


def test_context_features_come_from_meta_not_raw_cues() -> None:
    state = _sample_state(batch_size=1)
    context = require_context_summary(state)
    state.cue_packet = CuePacket(cues=torch.zeros(1, 4, 6), pooled=torch.zeros(1, 6))
    module = ReadoutChannels(ReadoutChannelsConfig(content_dim=6, channel_dim=8))
    features = module.extract_features(state)
    assert torch.allclose(features.context, context)


def test_format_readout_channels_report_is_human_readable() -> None:
    bundle, inspection = run_readout_channels_on_text(
        "Help bank!",
        num_traces=24,
        top_k=4,
        cue_dim=6,
        num_basins=8,
        settling_steps=2,
        channel_dim=8,
        seed=1,
    )
    _ = bundle
    report = format_readout_channels_report(inspection)
    assert "READOUT CHANNELS" in report
    assert "binding" in report
    assert "top basin pressures" in report


def test_readout_channels_cli_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/core/decode/readout_channels.py",
            "text",
            "Help bank!",
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
    assert "READOUT CHANNELS" in result.stdout
    assert "binding" in result.stdout


def test_run_readout_channels_on_text_smoke() -> None:
    bundle, inspection = run_readout_channels_on_text(
        "Help bank!",
        num_traces=24,
        top_k=4,
        cue_dim=6,
        num_basins=8,
        settling_steps=2,
        channel_dim=8,
        seed=1,
    )
    assert bundle.basin.shape == (1, 8)
    assert inspection.num_binding_edges >= 0
