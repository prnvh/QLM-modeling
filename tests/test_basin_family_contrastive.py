"""Commit D3: basin-family contrastive on attractor state + causal ablations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lmf.core.basin.basin_family_contrastive import (
    BasinFamilyContrastive,
    BasinFamilyContrastiveConfig,
    compute_basin_attractor_state,
    supervised_contrastive_loss,
)
from lmf.training.basin_families import BasinFamilyExample, load_basin_families
from lmf.training.basin_family_evaluator import evaluate_basin_families
from lmf.training.basin_family_overfit import run_basin_family_overfit_sanity
from lmf.training.basin_field_stack import build_basin_field_stack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "stage1" / "basin_families.jsonl"


def test_load_basin_families_validates_slug_and_coverage(tmp_path: Path) -> None:
    path = tmp_path / "families.jsonl"
    rows = [
        {"text": "one alpha", "family": "family-a"},
        {"text": "two alpha", "family": "family-a"},
        {"text": "one beta", "family": "family-b"},
        {"text": "two beta", "family": "family-b"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    examples = load_basin_families(path)
    assert len(examples) == 4


def test_load_basin_families_rejects_invalid_slug(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "hello", "family": "Bad Family"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="slug"):
        load_basin_families(path)


def test_attractor_state_uses_pressures_and_vectors() -> None:
    pressures = torch.tensor([[0.8, 0.1, 0.0], [0.0, 0.1, 0.9]], requires_grad=True)
    vectors = torch.randn(3, 4, requires_grad=True)
    state = compute_basin_attractor_state(pressures, vectors)
    assert state.shape == (2, 4)
    state.sum().backward()
    assert pressures.grad is not None
    assert vectors.grad is not None


def test_supervised_contrastive_prefers_same_family() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ]
    )
    family_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss_aligned = float(
        supervised_contrastive_loss(embeddings, family_index, temperature=0.1).item()
    )
    shuffled = embeddings[[0, 2, 1, 3]]
    loss_shuffled = float(
        supervised_contrastive_loss(shuffled, family_index, temperature=0.1).item()
    )
    assert loss_aligned < loss_shuffled


def test_basin_field_stack_gradients_reach_field_and_basin_bank() -> None:
    examples = [
        BasinFamilyExample("alpha beta", "family-a"),
        BasinFamilyExample("alpha again", "family-a"),
        BasinFamilyExample("gamma delta", "family-b"),
        BasinFamilyExample("gamma again", "family-b"),
    ]
    stack = build_basin_field_stack(examples, num_traces=24, top_k=8, num_basins=10, settling_steps=2)
    stack.train()
    loss, _metrics = stack.contrastive_loss_from_batch(examples)
    loss.backward()
    assert stack.field_loop.basin_bank.vectors.grad is not None
    assert stack.field_loop.binding_forces.basin_composer.pair_basin_support.projection[0].weight.grad is not None


def test_causal_ablations_hurt_separation_after_training() -> None:
    from lmf.training.basin_family_overfit import (
        default_overfit_examples,
        run_causal_ablation_margins,
        train_stack,
    )

    examples = default_overfit_examples()
    stack = build_basin_field_stack(
        examples,
        num_traces=32,
        top_k=12,
        num_basins=12,
        settling_steps=2,
        seed=7,
    )
    train_stack(stack, examples, steps=200, lr=5e-3, trace=False)
    full_margin = stack.separation_margin_for_examples(examples)
    no_bound_margin, permuted_margin = run_causal_ablation_margins(stack, examples)
    assert full_margin > no_bound_margin + 0.02
    assert full_margin > permuted_margin + 0.02


def test_evaluate_basin_families_untrained_has_pairs() -> None:
    if not DATA_PATH.is_file():
        pytest.skip("stage1 basin_families.jsonl not present")
    examples = load_basin_families(DATA_PATH)
    report = evaluate_basin_families(examples, num_traces=32, top_k=12, num_basins=12)
    assert report.num_pairs_same > 0
    assert report.num_pairs_different > 0


@pytest.mark.slow
def test_basin_family_overfit_sanity_gates_pass() -> None:
    result = run_basin_family_overfit_sanity()
    assert result.passed, result.failures
    assert result.no_bound_pair_margin < result.trained_report.separation_margin
    assert result.permuted_vectors_margin < result.trained_report.separation_margin


def test_basin_family_contrastive_cli_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/core/basin/basin_family_contrastive.py",
            "inspect",
            "--basin-dim",
            "8",
            "--trace",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pressures @ basin_bank.vectors" in result.stdout
    assert "basin_family_contrastive.forward" in result.stderr


def test_train_basin_families_cli_runs() -> None:
    if not DATA_PATH.is_file():
        pytest.skip("stage1 basin_families.jsonl not present")
    result = subprocess.run(
        [
            sys.executable,
            "lmf/training/train_basin_families.py",
            "--steps",
            "5",
            "--num-traces",
            "24",
            "--top-k",
            "8",
            "--num-basins",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "TRAINED SUMMARY" in result.stdout
