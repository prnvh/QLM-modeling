import logging
import subprocess
import sys

import pytest
import torch

from lmf.core.dmf.trace_bank import (
    TraceBank,
    TraceBankConfig,
    TraceBankState,
    build_trace_bank,
)


def test_trace_bank_exposes_learnable_parameters_with_expected_shapes() -> None:
    torch.manual_seed(2)
    bank = TraceBank(
        TraceBankConfig(
            num_traces=12,
            key_dim=5,
            content_dim=7,
        )
    )

    state = bank.state()

    assert isinstance(state, TraceBankState)
    assert state.keys.shape == (12, 5)
    assert state.content.shape == (12, 7)
    assert state.threshold.shape == (12,)
    assert state.decay.shape == (12,)
    assert torch.isfinite(state.keys).all()
    assert torch.isfinite(state.content).all()
    assert torch.isfinite(state.threshold).all()
    assert torch.isfinite(state.decay).all()
    assert (state.threshold > 0).all()
    assert ((state.decay > 0) & (state.decay < 1)).all()

    trainable = {name for name, parameter in bank.named_parameters()}
    assert trainable == {"keys", "content", "threshold", "decay"}


def test_trace_bank_gather_returns_selected_traces() -> None:
    bank = TraceBank(TraceBankConfig(num_traces=8, key_dim=4, content_dim=6))
    trace_ids = torch.tensor([1, 5, 3], dtype=torch.long)

    subset = bank.state(trace_ids=trace_ids)

    assert subset.keys.shape == (3, 4)
    assert subset.content.shape == (3, 6)
    assert subset.threshold.shape == (3,)
    assert subset.decay.shape == (3,)
    assert torch.allclose(subset.keys, bank.keys.index_select(0, trace_ids))
    assert torch.allclose(subset.content, bank.content.index_select(0, trace_ids))


def test_trace_bank_forward_matches_state() -> None:
    bank = TraceBank(TraceBankConfig(num_traces=4, key_dim=3, content_dim=3))
    trace_ids = torch.tensor([0, 2], dtype=torch.long)

    assert bank.forward().num_traces == bank.state().num_traces
    gathered_forward = bank.forward(trace_ids)
    gathered_state = bank.state(trace_ids=trace_ids)
    assert torch.allclose(gathered_forward.keys, gathered_state.keys)
    assert torch.allclose(gathered_forward.threshold, gathered_state.threshold)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TraceBankConfig(num_traces=0, key_dim=4, content_dim=4),
        lambda: TraceBankConfig(num_traces=4, key_dim=0, content_dim=4),
        lambda: TraceBankConfig(num_traces=4, key_dim=4, content_dim=0),
        lambda: TraceBankConfig(num_traces=4, key_dim=4, content_dim=4, init_scale=0.0),
        lambda: TraceBankConfig(num_traces=4, key_dim=4, content_dim=4, initial_threshold=0.0),
        lambda: TraceBankConfig(num_traces=4, key_dim=4, content_dim=4, initial_decay=0.0),
        lambda: TraceBankConfig(num_traces=4, key_dim=4, content_dim=4, initial_decay=1.0),
        lambda: TraceBankConfig(num_traces=4, key_dim=4, content_dim=4, trace_limit=-1),
    ],
)
def test_trace_bank_config_rejects_brittle_arguments(factory) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "trace_ids",
    [
        torch.empty(0, dtype=torch.long),
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.tensor([1.0, 2.0]),
        torch.tensor([0, 99], dtype=torch.long),
    ],
)
def test_trace_bank_rejects_invalid_trace_ids(trace_ids) -> None:
    bank = TraceBank(TraceBankConfig(num_traces=8, key_dim=4, content_dim=4))

    with pytest.raises(ValueError):
        bank.state(trace_ids=trace_ids)


def test_trace_bank_reset_parameters_restores_configured_scalars() -> None:
    bank = TraceBank(
        TraceBankConfig(
            num_traces=6,
            key_dim=4,
            content_dim=4,
            initial_threshold=0.25,
            initial_decay=0.2,
        )
    )
    with torch.no_grad():
        bank.keys.fill_(99.0)
        bank.content.fill_(-99.0)
        bank.threshold.fill_(0.0)
        bank.decay.fill_(0.0)

    bank.reset_parameters()
    state = bank.state()

    assert not torch.allclose(state.keys, torch.full_like(state.keys, 99.0))
    assert torch.allclose(state.threshold, torch.full_like(state.threshold, 0.25), atol=1e-5)
    assert torch.allclose(state.decay, torch.full_like(state.decay, 0.2), atol=1e-5)


def test_trace_bank_trace_logs_human_readable_state(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lmf.core.dmf.trace_bank")
    bank = TraceBank(
        TraceBankConfig(num_traces=6, key_dim=4, content_dim=4, trace=True, trace_limit=3)
    )

    bank.state(trace_ids=torch.tensor([0, 2, 4], dtype=torch.long))

    messages = [record.message for record in caplog.records]
    assert any("trace_bank.state" in message for message in messages)
    assert any("num_traces=3" in message for message in messages)
    assert any("key_shape=[3, 4]" in message for message in messages)


def test_build_trace_bank_is_reproducible_with_seed() -> None:
    first = build_trace_bank(num_traces=5, key_dim=3, content_dim=4, seed=21)
    second = build_trace_bank(num_traces=5, key_dim=3, content_dim=4, seed=21)

    assert torch.allclose(first.state().keys, second.state().keys)
    assert torch.allclose(first.state().content, second.state().content)


def test_trace_bank_file_can_be_run_directly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/core/dmf/trace_bank.py",
            "inspect",
            "--num-traces",
            "6",
            "--key-dim",
            "4",
            "--content-dim",
            "5",
            "--sample-ids",
            "0",
            "2",
            "4",
            "--seed",
            "11",
            "--trace",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Num traces: 6" in result.stdout
    assert "Keys shape: [6, 4]" in result.stdout
    assert "Content shape: [6, 5]" in result.stdout
    assert "Subset ids: [0, 2, 4]" in result.stdout
    assert "trace_bank.state" in result.stderr
