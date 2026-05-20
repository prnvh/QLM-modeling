"""Binding-gated interference with basin competition (Commit F).

Interference reads settled **basin pressures** and learnable basin vectors to
compute support, conflict, coexistence, and suppression among competing attractor
slots. Forces feed both trace drive and basin settling in ``FieldLoop``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.basin.basin_state import validate_basin_state  # noqa: E402
from lmf.core.dmf.trace_router import route_text  # noqa: E402
from lmf.core.field.context_op import ContextOp, ContextOpConfig  # noqa: E402
from lmf.core.state.types import ActiveRegion, BasinState, BindingState, InterferenceState  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _log_trace(logger: logging.Logger, enabled: bool, event: str, **fields: object) -> None:
    if not enabled:
        return
    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s%s", event, f" | {details}" if details else "")


@dataclass
class InterferenceLayerConfig:
    content_dim: int
    num_basins: int
    support_scale: float = 0.5
    conflict_scale: float = 1.0
    coexistence_scale: float = 0.35
    suppression_scale: float = 0.5
    trace: bool = False

    def __post_init__(self) -> None:
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        for name in ("support_scale", "conflict_scale", "coexistence_scale", "suppression_scale"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class InterferenceBreakdown:
    """Decomposed interference terms for logging, tests, and ablations."""

    trace_pair_energy: Tensor
    trace_local_energy: Tensor
    trace_contradiction: Tensor
    basin_support: Tensor
    basin_conflict: Tensor
    basin_coexistence: Tensor
    basin_suppression: Tensor
    basin_total_force: Tensor
    conflict_score: Tensor
    coexistence_score: Tensor
    interference_pressure: Tensor


def _binding_gate(binding_state: BindingState) -> Tensor:
    strengths = binding_state.relation_strength.clamp_min(0.0)
    if strengths.numel() == 0:
        batch_size = binding_state.edge_index.shape[0]
        return torch.zeros(batch_size, 1, device=binding_state.edge_index.device)
    return strengths.mean(dim=-1, keepdim=True)


def compute_basin_competition(
    basin_state: BasinState,
    *,
    binding_gate: Tensor,
    support_scale: float,
    conflict_scale: float,
    coexistence_scale: float,
    suppression_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute basin competition forces from pressures and attractor vectors."""

    validate_basin_state(basin_state)
    pressures = basin_state.pressures
    vectors = basin_state.vectors
    batch_size, num_basins = pressures.shape

    if vectors.shape[0] != num_basins:
        raise ValueError("basin_state.vectors rows must match pressures width.")

    dirs = F.normalize(vectors, dim=-1, eps=1e-6)
    similarity = dirs @ dirs.T
    similarity = similarity.clamp(-1.0, 1.0)

    co_activation = pressures.unsqueeze(-1) * pressures.unsqueeze(-2)
    eye = torch.eye(num_basins, device=pressures.device, dtype=torch.bool)
    co_activation = co_activation.masked_fill(eye.unsqueeze(0), 0.0)

    sim = similarity.unsqueeze(0).expand(batch_size, -1, -1)
    conflict_matrix = co_activation * (1.0 - sim).clamp_min(0.0)
    coexistence_matrix = co_activation * sim.clamp_min(0.0)

    conflict_per_basin = conflict_matrix.sum(dim=-1)
    coexistence_per_basin = coexistence_matrix.sum(dim=-1)

    top_pressure, _top_idx = pressures.max(dim=-1, keepdim=True)
    loser_mask = (pressures < top_pressure * 0.85).to(pressures.dtype)
    suppression_per_basin = conflict_per_basin * loser_mask

    gate = binding_gate.to(dtype=pressures.dtype)
    support_force = support_scale * coexistence_per_basin * gate
    conflict_force = conflict_scale * conflict_per_basin * gate
    coexistence_force = coexistence_scale * coexistence_per_basin * gate
    suppression_force = suppression_scale * suppression_per_basin * gate

    conflict_score = conflict_matrix.sum(dim=(1, 2)).unsqueeze(-1)
    coexistence_score = coexistence_matrix.sum(dim=(1, 2)).unsqueeze(-1)
    interference_pressure = (conflict_score + coexistence_score) * gate

    return (
        support_force,
        conflict_force,
        coexistence_force,
        suppression_force,
        conflict_score,
        coexistence_score,
        interference_pressure,
    )


class InterferenceLayer(nn.Module):
    """Learn compatibility/conflict energy terms gated by binding and basins."""

    def __init__(
        self,
        config: InterferenceLayerConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.logger = logger
        self.compatibility = nn.Parameter(torch.tensor(1.0))
        self.conflict = nn.Parameter(torch.tensor(0.25))

    def forward(
        self,
        active_region: ActiveRegion,
        binding_state: BindingState,
        basin_state: BasinState,
    ) -> InterferenceState:
        breakdown = self.compute_breakdown(active_region, binding_state, basin_state)
        return InterferenceState(
            pair_energy=breakdown.trace_pair_energy,
            local_energy=breakdown.trace_local_energy,
            contradiction=breakdown.trace_contradiction,
            basin_support_force=breakdown.basin_support,
            basin_conflict_force=breakdown.basin_conflict,
            basin_coexistence_force=breakdown.basin_coexistence,
            basin_suppression_force=breakdown.basin_suppression,
            conflict_score=breakdown.conflict_score,
            coexistence_score=breakdown.coexistence_score,
            interference_pressure=breakdown.interference_pressure,
        )

    def compute_breakdown(
        self,
        active_region: ActiveRegion,
        binding_state: BindingState,
        basin_state: BasinState,
    ) -> InterferenceBreakdown:
        content = active_region.trace_content
        if content.dim() != 3:
            raise ValueError("active_region.trace_content must have shape [batch, traces, dim].")

        binding_gate = _binding_gate(binding_state)
        normalized = F.normalize(content, dim=-1, eps=1e-6)
        pair_compat = torch.einsum("btd,btd->bt", normalized, normalized)
        local_energy = pair_compat.mean(dim=-1, keepdim=True)
        pair_energy = binding_gate * self.compatibility
        contradiction = (
            binding_gate
            * self.conflict
            * active_region.trace_amp.std(dim=-1, keepdim=True, unbiased=False)
        )

        (
            support,
            conflict,
            coexistence,
            suppression,
            conflict_score,
            coexistence_score,
            interference_pressure,
        ) = compute_basin_competition(
            basin_state,
            binding_gate=binding_gate,
            support_scale=self.config.support_scale,
            conflict_scale=self.config.conflict_scale,
            coexistence_scale=self.config.coexistence_scale,
            suppression_scale=self.config.suppression_scale,
        )

        basin_total = support + coexistence - conflict - suppression

        _log_trace(
            self.logger,
            self.config.trace,
            "interference.forward",
            conflict=float(conflict_score.mean().item()),
            coexistence=float(coexistence_score.mean().item()),
            basin_force_mean=float(basin_total.mean().item()),
        )

        return InterferenceBreakdown(
            trace_pair_energy=pair_energy,
            trace_local_energy=local_energy,
            trace_contradiction=contradiction,
            basin_support=support,
            basin_conflict=conflict,
            basin_coexistence=coexistence,
            basin_suppression=suppression,
            basin_total_force=basin_total,
            conflict_score=conflict_score,
            coexistence_score=coexistence_score,
            interference_pressure=interference_pressure,
        )


def compose_basin_interference_force(interference_state: InterferenceState) -> Tensor:
    """Signed basin force from decomposed interference terms."""

    required = (
        interference_state.basin_support_force,
        interference_state.basin_conflict_force,
        interference_state.basin_coexistence_force,
        interference_state.basin_suppression_force,
    )
    if any(value is None for value in required):
        raise ValueError("InterferenceState is missing basin competition forces.")
    return (
        required[0]
        + required[2]
        - required[1]
        - required[3]
    )


def run_interference_on_text(
    text: str,
    *,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    num_basins: int = 32,
    settling_steps: int = 1,
    seed: int = 7,
    trace: bool = False,
) -> InterferenceBreakdown:
    """Route text through one field step and return interference breakdown."""

    from lmf.core.field.loop import FieldLoop, FieldLoopConfig

    cue_packet, _routing, active_region = route_text(
        text,
        num_traces=num_traces,
        top_k=top_k,
        cue_dim=cue_dim,
        seed=seed,
    )
    field_loop = FieldLoop(
        FieldLoopConfig(
            cue_dim=cue_dim,
            content_dim=cue_dim,
            active_traces=top_k,
            num_basins=num_basins,
            settling_steps=settling_steps,
            trace=trace,
        )
    )
    context_op = ContextOp(
        ContextOpConfig(
            cue_dim=cue_dim,
            active_traces=top_k,
            num_basins=num_basins,
        )
    )
    basin_state = field_loop.make_basin_state(
        active_region.trace_amp.shape[0],
        device=active_region.trace_amp.device,
    )
    field_loop.eval()
    context_op.eval()

    with torch.no_grad():
        context = context_op(cue_packet, active_region)
        binding = field_loop.binding_layer(active_region, basin_state, context)
        return field_loop.interference_layer.compute_breakdown(active_region, binding, basin_state)


def format_interference_report(breakdown: InterferenceBreakdown, *, text: str) -> str:
    lines = [
        f"Input: {text}",
        f"trace_pair_energy: {float(breakdown.trace_pair_energy.mean().item()):.4f}",
        f"trace_contradiction: {float(breakdown.trace_contradiction.mean().item()):.4f}",
        f"conflict_score: {float(breakdown.conflict_score.mean().item()):.4f}",
        f"coexistence_score: {float(breakdown.coexistence_score.mean().item()):.4f}",
        f"interference_pressure: {float(breakdown.interference_pressure.mean().item()):.4f}",
        f"basin_total_force_mean: {float(breakdown.basin_total_force.mean().item()):.4f}",
    ]
    top_conflict = breakdown.basin_conflict[0].detach().cpu()
    top_indices = torch.topk(top_conflict, k=min(4, top_conflict.numel())).indices.tolist()
    if top_indices:
        lines.append("top conflict basins: " + ", ".join(f"basin_{idx}" for idx in top_indices))
    return "\n".join(lines)


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect basin-competition interference (Commit F).")
    parser.add_argument(
        "text",
        nargs="+",
        help='Text to process. You may optionally start with the word "text".',
    )
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--num-basins", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)
    breakdown = run_interference_on_text(
        text,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        num_basins=args.num_basins,
        seed=args.seed,
        trace=args.trace,
    )
    _safe_print(format_interference_report(breakdown, text=text))


__all__ = [
    "InterferenceBreakdown",
    "InterferenceLayer",
    "InterferenceLayerConfig",
    "compose_basin_interference_force",
    "compute_basin_competition",
    "format_interference_report",
    "run_interference_on_text",
]


if __name__ == "__main__":
    main()
