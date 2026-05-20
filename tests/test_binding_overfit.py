import subprocess
import sys

import pytest

from lmf.training.binding_edge_evaluator import evaluate_binding_edges
from lmf.training.binding_overfit import (
    OVERFIT_MAX_NEGATIVE_MASS,
    OVERFIT_MIN_ACCURACY,
    OVERFIT_MIN_POSITIVE_MASS,
    BindingOverfitConfig,
    default_overfit_examples,
    evaluate_overfit_gates,
    run_binding_overfit_sanity,
    train_binding_stack,
)
from lmf.training.binding_stack import build_binding_stack


def test_untrained_stack_fails_overfit_gates() -> None:
    examples = default_overfit_examples()
    stack = build_binding_stack(examples, cue_dim=16, num_traces=32, top_k=16, seed=7)
    untrained = evaluate_binding_edges(examples, stack=stack)
    result = evaluate_overfit_gates(untrained, untrained)

    assert not result.passed
    assert result.failures


@pytest.mark.slow
def test_binding_overfit_sanity_gates_pass() -> None:
    result = run_binding_overfit_sanity(
        config=BindingOverfitConfig(steps=400, seed=7),
        trace=False,
    )

    assert result.passed, result.failures
    assert result.full_report.binding_edge_accuracy >= OVERFIT_MIN_ACCURACY
    assert result.full_report.positive_binding_mass_mean >= OVERFIT_MIN_POSITIVE_MASS
    assert result.full_report.negative_binding_mass_mean <= OVERFIT_MAX_NEGATIVE_MASS
    assert result.full_report.binding_edge_accuracy > result.no_binding_report.binding_edge_accuracy


@pytest.mark.slow
def test_full_binding_beats_no_binding_ablation() -> None:
    examples = default_overfit_examples()
    config = BindingOverfitConfig(steps=400, seed=7)

    full_stack = train_binding_stack(examples, config, train_binding=True)
    no_binding_stack = train_binding_stack(examples, config, train_binding=False)

    full_report = evaluate_binding_edges(examples, stack=full_stack)
    no_binding_report = evaluate_binding_edges(examples, stack=no_binding_stack)

    assert full_report.binding_edge_accuracy > no_binding_report.binding_edge_accuracy
    assert full_report.positive_binding_mass_mean > no_binding_report.positive_binding_mass_mean


def test_binding_overfit_cli_runs_and_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/training/binding_overfit.py",
            "--steps",
            "400",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "BINDING OVERFIT SANITY (C5)" in result.stdout
    assert "passed=yes" in result.stdout
