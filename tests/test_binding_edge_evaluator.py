import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lmf.core.field.binding import BindingLayer, BindingLayerConfig
from lmf.core.field.binding_pair_scorer import BindingPairScorer, BindingPairScorerConfig
from lmf.core.field.types import ContextPressure
from lmf.core.input.cue_packet import CuePacket
from lmf.core.state.types import ActiveRegion, BasinState
from lmf.training.binding_edge_evaluator import evaluate_binding_edges, evaluate_binding_edges_on_example
from lmf.training.binding_edges import BindingEdge, BindingEdgeExample, load_binding_edges, resolve_example_edges
from lmf.training.binding_stack import BindingStack, BindingStackConfig, build_binding_stack

DATA_PATH = Path("data/stage1/binding_edges.jsonl")


def _placeholder_basin(active_region: ActiveRegion) -> BasinState:
    return BasinState(
        pressures=torch.zeros(active_region.trace_amp.shape[0], 1),
        vectors=torch.zeros(1, active_region.trace_content.shape[-1]),
    )


def _context(*, batch: int = 1, traces: int = 3, cue_dim: int = 4) -> ContextPressure:
    return ContextPressure(
        trace_drive=torch.zeros(batch, traces),
        basin_drive=torch.zeros(batch, 1),
        threshold_shift=torch.zeros(batch, traces),
        context_summary=torch.randn(batch, cue_dim),
    )


def test_binding_pair_scorer_changes_with_span_distance() -> None:
    scorer = BindingPairScorer(BindingPairScorerConfig(content_dim=4, context_dim=4, num_relations=2))
    content = torch.randn(1, 2, 4)
    context = torch.randn(1, 4)

    near = ActiveRegion(
        trace_ids=torch.tensor([[0, 1]]),
        trace_content=content,
        trace_amp=torch.ones(1, 2),
        cue_drive=torch.zeros(1, 2),
        source_cue_id=torch.tensor([[1, 2]]),
        source_span=torch.tensor([[[1, 1], [2, 2]]]),
        cue_type=torch.tensor([[5, 5]]),
    )
    far = ActiveRegion(
        trace_ids=torch.tensor([[0, 1]]),
        trace_content=content,
        trace_amp=torch.ones(1, 2),
        cue_drive=torch.zeros(1, 2),
        source_cue_id=torch.tensor([[1, 9]]),
        source_span=torch.tensor([[[1, 1], [9, 9]]]),
        cue_type=torch.tensor([[5, 5]]),
    )
    _ = far

    _, order_near = scorer._pair_span_features(near, num_traces=2)
    swapped = ActiveRegion(
        trace_ids=torch.tensor([[0, 1]]),
        trace_content=content,
        trace_amp=torch.ones(1, 2),
        cue_drive=torch.zeros(1, 2),
        source_cue_id=torch.tensor([[2, 1]]),
        source_span=torch.tensor([[[2, 2], [1, 1]]]),
        cue_type=torch.tensor([[5, 5]]),
    )
    _, order_swapped = scorer._pair_span_features(swapped, num_traces=2)
    assert float(order_near[0, 0, 1].item()) != float(order_swapped[0, 0, 1].item())


def test_pairwise_strength_matrix_uses_context() -> None:
    layer = BindingLayer(BindingLayerConfig(content_dim=4, context_dim=4))
    active_region = ActiveRegion(
        trace_ids=torch.tensor([[0, 1, 2]]),
        trace_content=torch.randn(1, 3, 4),
        trace_amp=torch.rand(1, 3),
        cue_drive=torch.randn(1, 3),
        mask=torch.ones(1, 3, dtype=torch.bool),
        source_cue_id=torch.tensor([[0, 1, 2]]),
        source_span=torch.tensor([[[0, 0], [1, 1], [2, 2]]]),
        cue_type=torch.tensor([[5, 5, 5]]),
    )
    ctx_a = _context(traces=3, cue_dim=4)
    ctx_b = _context(traces=3, cue_dim=4)

    mass_a = layer.pairwise_strength_matrix(active_region, context=ctx_a)
    mass_b = layer.pairwise_strength_matrix(active_region, context=ctx_b)

    assert mass_a.shape == (1, 3, 3)
    assert not torch.allclose(mass_a, mass_b)


def test_resolve_example_edges_uses_sequence_positions() -> None:
    example = BindingEdgeExample(
        text="help bank",
        edges=(BindingEdge("help", "bank", 1),),
    )
    resolved = resolve_example_edges(example)[0]
    assert resolved.cue_a_positions
    assert resolved.cue_b_positions
    assert resolved.cue_a_positions != resolved.cue_b_positions


def test_binding_stack_forward_example_end_to_end() -> None:
    examples = load_binding_edges(DATA_PATH)
    stack = build_binding_stack(examples[:1], cue_dim=8, num_traces=24, top_k=8, seed=2)
    stack.eval()
    with torch.no_grad():
        result = stack.forward_example(examples[0])
    assert result.pair_mass.shape == (1, 8, 8)
    assert len(result.resolved_edges) == len(examples[0].edges)


def test_evaluate_binding_edges_on_example_returns_metrics() -> None:
    examples = load_binding_edges(DATA_PATH)
    stack = build_binding_stack(examples, cue_dim=8, num_traces=24, top_k=12, seed=3)
    report = evaluate_binding_edges_on_example(examples[0], stack=stack)
    assert report.num_edges == len(examples[0].edges)
    assert 0.0 <= report.binding_edge_accuracy <= 1.0


def test_binding_stack_loss_backprops() -> None:
    from lmf.training.binding_edges import ResolvedBindingEdge
    from lmf.training.binding_stack import BindingForwardResult

    examples = [
        BindingEdgeExample(text="help bank", edges=(BindingEdge("help", "bank", 1),)),
    ]
    stack = build_binding_stack(examples, cue_dim=4, num_traces=16, top_k=4, seed=1)
    content = torch.randn(1, 2, 4, requires_grad=True)
    active_region = ActiveRegion(
        trace_ids=torch.tensor([[0, 1]]),
        trace_content=content,
        trace_amp=torch.ones(1, 2),
        cue_drive=torch.zeros(1, 2),
        source_cue_id=torch.tensor([[1, 2]]),
        source_span=torch.tensor([[[1, 1], [2, 2]]]),
        cue_type=torch.tensor([[5, 5]]),
    )
    context = _context(traces=2, cue_dim=4)
    pair_mass = stack.binding_layer.pairwise_strength_matrix(active_region, context=context)
    result = BindingForwardResult(
        text=examples[0].text,
        cue_packet=CuePacket(cues=torch.randn(1, 4, 4), pooled=torch.randn(1, 4)),
        active_region=active_region,
        context=context,
        pair_mass=pair_mass,
        binding_state=stack.binding_layer(active_region, _placeholder_basin(active_region), context),
        resolved_edges=(ResolvedBindingEdge("help", "bank", 1, (1,), (2,)),),
    )
    loss = stack.binding_edge_loss(result)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert content.grad is not None


def test_train_binding_edges_improves_small_set() -> None:
    examples = [
        BindingEdgeExample(
            text="help bank",
            edges=(
                BindingEdge("help", "bank", 1),
                BindingEdge("help", "help", 0),
            ),
        )
    ]
    stack = build_binding_stack(examples, cue_dim=8, num_traces=32, top_k=8, seed=4)
    optimizer = torch.optim.Adam(stack.parameters(), lr=5e-3)

    before = evaluate_binding_edges_on_example(examples[0], stack=stack)
    for _ in range(80):
        stack.train()
        optimizer.zero_grad()
        result = stack.forward_example(examples[0])
        loss = stack.binding_edge_loss(result)
        loss.backward()
        optimizer.step()
    after = evaluate_binding_edges_on_example(examples[0], stack=stack)

    assert after.binding_edge_loss <= before.binding_edge_loss
    assert after.positive_binding_mass_mean >= before.positive_binding_mass_mean


def test_binding_edge_evaluator_cli_runs_on_dataset() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/training/binding_edge_evaluator.py",
            "--num-traces",
            "24",
            "--top-k",
            "12",
            "--cue-dim",
            "8",
            "--seed",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DATASET" in result.stdout
    assert "BINDING EDGE EVAL" in result.stdout


def test_train_binding_edges_cli_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/training/train_binding_edges.py",
            "--steps",
            "5",
            "--top-k",
            "8",
            "--cue-dim",
            "6",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "TRAINED SUMMARY" in result.stdout
