import logging
import subprocess
import sys

import pytest
import torch
from torch import nn

from lmf.core.dmf.trace_bank import TraceBank, TraceBankConfig
from lmf.core.field.loop import FieldLoop, FieldLoopConfig
from lmf.training.module_diagnostics import (
    ModuleDiagnosticReport,
    collect_module_diagnostics,
    format_module_diagnostics,
    gradient_norm,
    log_module_diagnostics,
    parameter_count,
    run_diagnostics_demo,
)


class _FieldOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trace_bank = TraceBank(TraceBankConfig(num_traces=12, key_dim=4, content_dim=4))
        self.field_loop = FieldLoop(
            FieldLoopConfig(cue_dim=4, content_dim=4, active_traces=3, num_basins=5, settling_steps=1)
        )


def test_parameter_count_sums_module_weights() -> None:
    module = _FieldOnlyModel()
    assert parameter_count(module.field_loop) == sum(p.numel() for p in module.field_loop.parameters())
    assert parameter_count(module.field_loop) > 0


def test_gradient_norm_is_none_before_backward() -> None:
    module = _FieldOnlyModel()
    assert gradient_norm(module.field_loop.binding_layer) is None


def test_collect_module_diagnostics_reports_b3_fields_after_backward() -> None:
    torch.manual_seed(1)
    module = _FieldOnlyModel()
    for param in module.parameters():
        param.grad = None

    loss = sum(param.sum() for param in module.parameters())
    loss.backward()

    report = collect_module_diagnostics(module)

    assert isinstance(report, ModuleDiagnosticReport)
    assert report.trace_bank_grad_norm is not None and report.trace_bank_grad_norm > 0
    assert report.binding_layer_grad_norm is not None and report.binding_layer_grad_norm > 0
    assert report.binding_forces_grad_norm is not None and report.binding_forces_grad_norm > 0
    assert report.interference_grad_norm is not None and report.interference_grad_norm > 0
    assert report.decoder_grad_norm is None
    assert report.field_loop_param_count > 0
    assert report.binding_param_count > 0
    assert report.interference_param_count > 0


def test_binding_param_count_includes_layer_and_forces() -> None:
    module = _FieldOnlyModel()
    report = collect_module_diagnostics(module)

    expected = parameter_count(module.field_loop.binding_layer) + parameter_count(
        module.field_loop.binding_forces
    )
    assert report.binding_param_count == expected


def test_format_module_diagnostics_is_human_readable() -> None:
    report = ModuleDiagnosticReport(
        trace_bank_grad_norm=0.5,
        binding_layer_grad_norm=0.25,
        binding_forces_grad_norm=0.1,
        interference_grad_norm=0.05,
        basin_grad_norm=None,
        decoder_grad_norm=None,
        field_loop_param_count=123,
        binding_param_count=45,
        interference_param_count=6,
    )
    text = format_module_diagnostics(report)

    assert "trace_bank_grad_norm=0.500000" in text
    assert "basin_grad_norm=n/a" in text
    assert "field_loop_param_count=123" in text


def test_log_module_diagnostics_emits_expected_keys(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lmf.training.module_diagnostics")
    module = _FieldOnlyModel()
    sum(param.sum() for param in module.parameters()).backward()

    log_module_diagnostics(module, step=3)

    message = " ".join(record.message for record in caplog.records)
    assert "module_diagnostics" in message
    assert "step=3" in message
    assert "binding_layer_grad_norm=" in message
    assert "field_loop_param_count=" in message


def test_run_diagnostics_demo_produces_finite_grad_norms() -> None:
    report, model = run_diagnostics_demo(
        "Help bank!",
        num_traces=24,
        top_k=4,
        cue_dim=6,
        num_basins=8,
        settling_steps=1,
        seed=5,
    )

    assert report.binding_layer_grad_norm is not None
    assert report.decoder_grad_norm is not None
    assert report.decoder_grad_norm > 0
    assert parameter_count(model.decoder) > 0


def test_module_diagnostics_cli_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/training/module_diagnostics.py",
            "text",
            "Help bank!",
            "--num-traces",
            "24",
            "--top-k",
            "4",
            "--cue-dim",
            "6",
            "--seed",
            "3",
            "--trace",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Input: Help bank!" in result.stdout
    assert "binding_layer_grad_norm=" in result.stdout
    assert "field_loop_param_count=" in result.stdout
    assert "module_diagnostics" in result.stderr
