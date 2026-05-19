import logging
import subprocess
import sys

import pytest
import torch

from lmf.core.field.loop import (
    FieldLoop,
    FieldLoopConfig,
    make_placeholder_basin_state,
    run_field_loop_on_text,
)
from lmf.core.input.cue_packet import CuePacket
from lmf.core.state.types import ActiveRegion


def _sample_active_region(*, batch_size: int = 2, num_traces: int = 5, dim: int = 6) -> ActiveRegion:
    return ActiveRegion(
        trace_ids=torch.arange(num_traces).unsqueeze(0).expand(batch_size, num_traces),
        trace_content=torch.randn(batch_size, num_traces, dim),
        trace_amp=torch.rand(batch_size, num_traces),
        cue_drive=torch.randn(batch_size, num_traces),
        mask=torch.ones(batch_size, num_traces, dtype=torch.bool),
    )


def _sample_cue_packet(*, batch_size: int = 2, seq_len: int = 4, cue_dim: int = 6) -> CuePacket:
    return CuePacket(
        cues=torch.randn(batch_size, seq_len, cue_dim),
        mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
        pooled=torch.randn(batch_size, cue_dim),
    )


def test_field_loop_forward_runs_settling_steps() -> None:
    torch.manual_seed(3)
    field_loop = FieldLoop(
        FieldLoopConfig(
            cue_dim=6,
            content_dim=6,
            active_traces=5,
            num_basins=8,
            settling_steps=3,
        )
    )
    cue_packet = _sample_cue_packet()
    active_region = _sample_active_region()
    basin_state = make_placeholder_basin_state(
        batch_size=2,
        num_basins=8,
        basin_dim=6,
        device=active_region.trace_amp.device,
        dtype=active_region.trace_amp.dtype,
    )

    output = field_loop(cue_packet, active_region, basin_state)

    assert output.steps_run == 3
    assert output.active_region_trace_amp.shape == (2, 5)
    assert output.basin_pressures.shape == (2, 8)
    assert torch.isfinite(output.active_region_trace_amp).all()
    assert torch.isfinite(output.basin_pressures).all()


def test_field_loop_forward_supports_gradients() -> None:
    field_loop = FieldLoop(
        FieldLoopConfig(cue_dim=4, content_dim=4, active_traces=3, num_basins=5, settling_steps=2)
    )
    cue_packet = _sample_cue_packet(batch_size=1, cue_dim=4)
    active_region = _sample_active_region(batch_size=1, num_traces=3, dim=4)
    basin_state = make_placeholder_basin_state(
        batch_size=1,
        num_basins=5,
        basin_dim=4,
        device=active_region.trace_amp.device,
        dtype=active_region.trace_amp.dtype,
    )

    output = field_loop(cue_packet, active_region, basin_state)
    loss = output.active_region_trace_amp.sum() + output.basin_pressures.sum()
    loss.backward()

    assert any(param.grad is not None for param in field_loop.parameters())


def test_field_loop_trace_logs_steps(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lmf.core.field.loop")
    field_loop = FieldLoop(
        FieldLoopConfig(
            cue_dim=4,
            content_dim=4,
            active_traces=3,
            num_basins=5,
            settling_steps=2,
            trace=True,
        )
    )
    field_loop(
        _sample_cue_packet(batch_size=1, cue_dim=4),
        _sample_active_region(batch_size=1, num_traces=3, dim=4),
        make_placeholder_basin_state(
            batch_size=1,
            num_basins=5,
            basin_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
    )

    messages = [record.message for record in caplog.records]
    assert any("field_loop.step" in message for message in messages)


def test_field_loop_file_can_be_run_directly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/core/field/loop.py",
            "text",
            "Help bank!",
            "--top-k",
            "5",
            "--cue-dim",
            "6",
            "--settling-steps",
            "2",
            "--seed",
            "11",
            "--trace",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Input: Help bank!" in result.stdout
    assert "Steps: 2" in result.stdout
    assert "field_loop.step" in result.stderr


def test_run_field_loop_on_text_end_to_end() -> None:
    output = run_field_loop_on_text("Help bank!", top_k=4, cue_dim=6, settling_steps=2, seed=9)

    assert output.steps_run == 2
    assert output.active_region_trace_amp.shape == (1, 4)
