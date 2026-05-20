"""Train context-sensitive binding on labeled cue-pair edges."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.training.binding_edge_evaluator import (  # noqa: E402
    evaluate_binding_edges,
    format_binding_edge_report,
)
from lmf.training.binding_edges import load_binding_edges  # noqa: E402
from lmf.training.binding_stack import BindingStack, build_binding_stack  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_BINDING_EDGES_PATH = PROJECT_ROOT / "data" / "stage1" / "binding_edges.jsonl"


def train_binding_edges(
    *,
    edges_file: str | Path,
    steps: int = 200,
    lr: float = 1e-3,
    num_traces: int = 64,
    top_k: int = 16,
    cue_dim: int = 16,
    seed: int = 7,
    trace: bool = False,
) -> None:
    examples = load_binding_edges(edges_file)
    stack = build_binding_stack(
        examples,
        cue_dim=cue_dim,
        num_traces=num_traces,
        top_k=top_k,
        seed=seed,
    )
    optimizer = torch.optim.Adam(stack.parameters(), lr=lr)

    for step in range(1, steps + 1):
        stack.train()
        optimizer.zero_grad()
        step_losses: list[Tensor] = []
        for example in examples:
            result = stack.forward_example(example)
            loss = stack.binding_edge_loss(result)
            if float(loss.detach()) > 0:
                step_losses.append(loss)
        if step_losses:
            total = torch.stack(step_losses).mean()
            total.backward()
            optimizer.step()
            mean_loss = float(total.detach().item())
        else:
            mean_loss = 0.0
        if trace or step == 1 or step == steps or step % max(steps // 5, 1) == 0:
            report = evaluate_binding_edges(examples, stack=stack)
            LOGGER.info(
                "train_binding step=%s loss=%.4f acc=%.4f pos_mass=%.4f neg_mass=%.4f",
                step,
                mean_loss,
                report.binding_edge_accuracy,
                report.positive_binding_mass_mean,
                report.negative_binding_mass_mean,
            )

    report = evaluate_binding_edges(examples, stack=stack)
    print(format_binding_edge_report(text="TRAINED SUMMARY", report=report))


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binding stack on cue-pair edge labels.")
    parser.add_argument("--edges-file", type=str, default=str(DEFAULT_BINDING_EDGES_PATH))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if args.trace:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    train_binding_edges(
        edges_file=args.edges_file,
        steps=args.steps,
        lr=args.lr,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        seed=args.seed,
        trace=args.trace,
    )


if __name__ == "__main__":
    main()
