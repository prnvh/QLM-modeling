"""Commit E: channelized cognitive-state decoder + same-checkpoint ablations."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
import yaml

from lmf.core.decode.decoder import (
    STANDARD_DECODER_ABLATIONS,
    TRACE_ONLY_ABLATION,
    ZERO_BINDING_ABLATION,
    ZERO_BASINS_ABLATION,
    ChannelAblation,
    CognitiveStateDecoder,
    DecoderConfig,
    ReadoutDropoutConfig,
    decoder_config_from_mapping,
    run_decoder_on_text,
)
from lmf.core.field.loop import FieldLoop, FieldLoopConfig
from lmf.core.state.cognitive_state import (
    build_cognitive_state,
    build_cognitive_state_from_field_loop,
    require_context_summary,
)
from lmf.core.state.types import (
    ActiveRegion,
    BasinState,
    BindingState,
    CognitiveState,
    CuePacket,
    InterferenceState,
    LucidityState,
)
from lmf.training.decoder_ablation_evaluator import (
    build_decoder_eval_stack,
    cosine_distance,
    evaluate_decoder_ablations,
    evaluate_decoder_ablations_on_text,
    evaluate_settling_k_ablation,
    format_decoder_ablation_report,
)


def _sample_state(*, batch_size: int = 2, num_traces: int = 5, dim: int = 6) -> CognitiveState:
    cue_packet = CuePacket(
        cues=torch.randn(batch_size, 4, dim),
        mask=torch.ones(batch_size, 4, dtype=torch.bool),
        pooled=torch.randn(batch_size, dim),
    )
    active_region = ActiveRegion(
        trace_ids=torch.arange(num_traces).unsqueeze(0).expand(batch_size, num_traces),
        trace_content=torch.randn(batch_size, num_traces, dim),
        trace_amp=torch.rand(batch_size, num_traces),
        cue_drive=torch.randn(batch_size, num_traces),
        mask=torch.ones(batch_size, num_traces, dtype=torch.bool),
    )
    basin_state = BasinState(
        pressures=torch.rand(batch_size, 8),
        vectors=torch.randn(8, dim),
    )
    binding_state = BindingState(
        edge_index=torch.stack(
            [
                torch.arange(num_traces).repeat_interleave(2).unsqueeze(0).expand(batch_size, -1),
                torch.arange(num_traces).repeat_interleave(2).roll(1).unsqueeze(0).expand(batch_size, -1),
            ],
            dim=1,
        ),
        relation_strength=torch.rand(batch_size, num_traces * 2),
        relation_index=torch.zeros(batch_size, num_traces * 2, dtype=torch.long),
        centrality=torch.softmax(torch.rand(batch_size, num_traces), dim=-1),
    )
    interference_state = InterferenceState(
        pair_energy=torch.rand(batch_size, 1),
        local_energy=torch.rand(batch_size, 1),
        contradiction=torch.rand(batch_size, 1),
    )
    lucidity_state = LucidityState(
        score=torch.rand(batch_size, 1),
        stability=torch.rand(batch_size, 1),
        ambiguity=torch.rand(batch_size, 1),
    )
    return build_cognitive_state(
        cue_packet=cue_packet,
        active_region=active_region,
        basin_state=basin_state,
        binding_state=binding_state,
        interference_state=interference_state,
        lucidity_state=lucidity_state,
        context_summary=torch.randn(batch_size, dim),
    )


def test_build_cognitive_state_stores_context_summary() -> None:
    state = _sample_state(batch_size=1)
    summary = require_context_summary(state)
    assert summary.shape == (1, 6)


def test_decoder_outputs_normalized_readout_and_channels() -> None:
    torch.manual_seed(0)
    state = _sample_state()
    decoder = CognitiveStateDecoder(DecoderConfig(content_dim=6, readout_dim=8))
    output = decoder(state)

    assert output.readout.shape == (2, 8)
    assert torch.isfinite(output.readout).all()
    assert torch.allclose(output.readout.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert set(output.channel_vectors) == {
        "trace",
        "binding",
        "basin",
        "interference",
        "lucidity",
        "context",
    }


def test_decoder_rejects_raw_cue_shortcut_flag() -> None:
    state = _sample_state(batch_size=1)
    state.meta["raw_cue_shortcut"] = True
    decoder = CognitiveStateDecoder(
        DecoderConfig(content_dim=6, readout_dim=8, no_direct_cue_shortcut=True)
    )
    with pytest.raises(ValueError, match="raw cue shortcuts"):
        decoder(state)


def test_decoder_requires_context_summary() -> None:
    state = _sample_state(batch_size=1)
    state.meta.pop("context_summary")
    decoder = CognitiveStateDecoder(DecoderConfig(content_dim=6, readout_dim=8))
    with pytest.raises(ValueError, match="context_summary"):
        decoder(state)


def test_channel_scales_change_readout() -> None:
    torch.manual_seed(1)
    state = _sample_state(batch_size=1)
    low_scale = CognitiveStateDecoder(
        DecoderConfig(content_dim=6, readout_dim=8, basin_channel_scale=0.0)
    )
    high_scale = CognitiveStateDecoder(
        DecoderConfig(content_dim=6, readout_dim=8, basin_channel_scale=2.0)
    )
    low_scale.load_state_dict(high_scale.state_dict())

    low = low_scale(state).readout
    high = high_scale(state).readout
    assert not torch.allclose(low, high)


def test_trace_only_ablation_differs_from_full() -> None:
    torch.manual_seed(2)
    state = _sample_state(batch_size=1)
    decoder = CognitiveStateDecoder(DecoderConfig(content_dim=6, readout_dim=8))
    full = decoder(state).readout
    trace_only = decoder(state, ablation=TRACE_ONLY_ABLATION).readout
    assert cosine_distance(full, trace_only) > 0.05


def test_zero_binding_and_zero_basins_change_readout() -> None:
    torch.manual_seed(3)
    state = _sample_state(batch_size=1)
    decoder = CognitiveStateDecoder(DecoderConfig(content_dim=6, readout_dim=8))
    full = decoder(state).readout
    no_binding = decoder(state, ablation=ZERO_BINDING_ABLATION).readout
    no_basins = decoder(state, ablation=ZERO_BASINS_ABLATION).readout
    assert cosine_distance(full, no_binding) > 0.01
    assert cosine_distance(full, no_basins) > 0.01


def test_decoder_supports_gradients() -> None:
    state = _sample_state(batch_size=1)
    decoder = CognitiveStateDecoder(DecoderConfig(content_dim=6, readout_dim=8))
    output = decoder(state)
    output.readout.sum().backward()
    assert decoder.readout_channels.basin_projector.net[0].weight.grad is not None


def test_decoder_config_from_yaml_mapping() -> None:
    raw = {
        "trace_channel_scale": 0.5,
        "basin_channel_scale": 1.5,
        "readout_dropout": {"trace": 0.1, "binding": 0.2, "basin": 0.2, "interference": 0.15},
    }
    config = decoder_config_from_mapping(raw, content_dim=16, readout_dim=32)
    assert config.trace_channel_scale == 0.5
    assert config.basin_channel_scale == 1.5
    assert config.readout_dropout.binding == 0.2


def test_stage1_yaml_contains_decoder_section() -> None:
    path = "lmf/infra/config/stage1_local.yaml"
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["decoder"]["trace_channel_scale"] == 0.7
    assert config["readout_dropout"]["binding"] == 0.10


def test_evaluate_decoder_ablations_reports_cosine_deltas() -> None:
    torch.manual_seed(4)
    state = _sample_state(batch_size=1)
    decoder = CognitiveStateDecoder(DecoderConfig(content_dim=6, readout_dim=8))
    records = evaluate_decoder_ablations(state, decoder)
    names = [record.name for record in records]
    assert names == list(STANDARD_DECODER_ABLATIONS)
    full = next(record for record in records if record.name == "full")
    trace_only = next(record for record in records if record.name == "trace_only")
    assert full.cosine_to_full == 0.0
    assert trace_only.cosine_to_full is not None and trace_only.cosine_to_full > 0.0


def test_evaluate_decoder_ablations_on_text_runs() -> None:
    field_loop, context_op, decoder = build_decoder_eval_stack(
        num_traces=24,
        top_k=4,
        cue_dim=6,
        num_basins=8,
        settling_steps=2,
        readout_dim=8,
    )
    report = evaluate_decoder_ablations_on_text(
        "Help me withdraw money from the bank",
        decoder=decoder,
        field_loop=field_loop,
        context_op=context_op,
        num_traces=24,
        top_k=4,
        cue_dim=6,
        settling_steps=2,
        seed=3,
    )
    text = format_decoder_ablation_report(report)
    assert "trace_only" in text
    assert "zero_binding" in text


def test_settling_k_ablation_can_differ() -> None:
    field_loop, context_op, decoder = build_decoder_eval_stack(
        num_traces=24,
        top_k=4,
        cue_dim=6,
        num_basins=8,
        settling_steps=3,
        readout_dim=8,
    )
    k_full, k_one = evaluate_settling_k_ablation(
        "Help bank!",
        decoder=decoder,
        field_loop=field_loop,
        context_op=context_op,
        num_traces=24,
        top_k=4,
        cue_dim=6,
        full_steps=3,
        reduced_steps=1,
        seed=2,
    )
    assert k_full.name == "K=3"
    assert k_one.name == "K=1"
    assert k_one.cosine_to_full is not None


def test_build_cognitive_state_from_field_loop_merges_pressures() -> None:
    torch.manual_seed(5)
    field_loop = FieldLoop(
        FieldLoopConfig(cue_dim=4, content_dim=4, active_traces=3, num_basins=5, settling_steps=1)
    )
    cue_packet = CuePacket(cues=torch.randn(1, 3, 4), pooled=torch.randn(1, 4))
    active_region = ActiveRegion(
        trace_ids=torch.tensor([[0, 1, 2]]),
        trace_content=torch.randn(1, 3, 4),
        trace_amp=torch.rand(1, 3),
        cue_drive=torch.randn(1, 3),
    )
    basin_state = field_loop.make_basin_state(1)
    field_output = field_loop(cue_packet, active_region, basin_state)
    state = build_cognitive_state_from_field_loop(
        cue_packet=cue_packet,
        active_region=active_region,
        basin_state=basin_state,
        field_output=field_output,
        context_summary=torch.randn(1, 4),
    )
    assert torch.allclose(state.active_region.trace_amp, field_output.active_region_trace_amp)
    assert torch.allclose(state.basin_state.pressures, field_output.basin_pressures)


def test_readout_dropout_config_validates_range() -> None:
    with pytest.raises(ValueError, match="readout_dropout"):
        ReadoutDropoutConfig(trace=1.0)


def test_decoder_cli_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/core/decode/decoder.py",
            "text",
            "Help bank!",
            "--num-traces",
            "24",
            "--top-k",
            "4",
            "--cue-dim",
            "6",
            "--ablation",
            "trace_only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Ablation: trace_only" in result.stdout
    assert "Readout shape:" in result.stdout


def test_decoder_ablation_evaluator_cli_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/training/decoder_ablation_evaluator.py",
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
    assert "zero_binding" in result.stdout
    assert "settling ablation" in result.stdout


def test_run_decoder_on_text_smoke() -> None:
    output = run_decoder_on_text(
        "Help bank!",
        num_traces=24,
        top_k=4,
        cue_dim=6,
        num_basins=8,
        settling_steps=2,
        readout_dim=8,
        seed=1,
    )
    assert output.readout.shape == (1, 8)
