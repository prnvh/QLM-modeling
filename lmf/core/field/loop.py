"""Recurrent field loop: context, binding, forces, interference, settling."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.basin.basin_bank import BasinBank, BasinBankConfig  # noqa: E402
from lmf.core.basin.basin_state import BasinStateSpec, make_basin_state, validate_basin_state  # noqa: E402
from lmf.core.dmf.trace_router import route_text  # noqa: E402
from lmf.core.field.binding import BindingLayer, BindingLayerConfig  # noqa: E402
from lmf.core.field.binding_forces import BindingForcesConfig, BindingForcesModule  # noqa: E402
from lmf.core.field.context_op import ContextOp, ContextOpConfig  # noqa: E402
from lmf.core.field.interference import InterferenceLayer, InterferenceLayerConfig, compose_basin_interference_force  # noqa: E402
from lmf.core.field.settling import Settling, SettlingConfig  # noqa: E402
from lmf.core.field.types import FieldLoopOutput  # noqa: E402
from lmf.core.input.cue_packet import CuePacket  # noqa: E402
from lmf.core.state.types import ActiveRegion, BasinState, BindingState, InterferenceState  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _log_trace(logger: logging.Logger, enabled: bool, event: str, **fields: object) -> None:
    if not enabled:
        return
    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s%s", event, f" | {details}" if details else "")


@dataclass
class FieldLoopConfig:
    cue_dim: int
    content_dim: int
    active_traces: int
    num_basins: int
    relation_channels: int = 4
    settling_steps: int = 3
    trace: bool = False

    def __post_init__(self) -> None:
        if self.cue_dim <= 0:
            raise ValueError("cue_dim must be positive.")
        if self.content_dim <= 0:
            raise ValueError("content_dim must be positive.")
        if self.active_traces <= 0:
            raise ValueError("active_traces must be positive.")
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.relation_channels <= 0:
            raise ValueError("relation_channels must be positive.")
        if self.settling_steps <= 0:
            raise ValueError("settling_steps must be positive.")


def make_placeholder_basin_state(
    *,
    batch_size: int,
    num_basins: int,
    basin_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BasinState:
    """Build a validated zero-pressure basin state (non-learned vectors)."""

    return make_basin_state(
        batch_size=batch_size,
        num_basins=num_basins,
        basin_dim=basin_dim,
        device=device,
        dtype=dtype,
    )


class FieldLoop(nn.Module):
    """Owns trainable field submodules and runs sparse settling steps."""

    def __init__(
        self,
        config: FieldLoopConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.logger = logger

        self.context_op = ContextOp(
            ContextOpConfig(
                cue_dim=config.cue_dim,
                active_traces=config.active_traces,
                num_basins=config.num_basins,
            )
        )
        self.binding_layer = BindingLayer(
            BindingLayerConfig(
                content_dim=config.content_dim,
                context_dim=config.cue_dim,
                relation_channels=config.relation_channels,
            )
        )
        self.binding_forces = BindingForcesModule(
            BindingForcesConfig(
                num_basins=config.num_basins,
                content_dim=config.content_dim,
                num_relations=config.relation_channels,
            )
        )
        self.basin_bank = BasinBank(
            BasinBankConfig(
                num_basins=config.num_basins,
                basin_dim=config.content_dim,
            )
        )
        self.interference_layer = InterferenceLayer(
            InterferenceLayerConfig(
                content_dim=config.content_dim,
                num_basins=config.num_basins,
                trace=config.trace,
            )
        )
        self.settling = Settling(SettlingConfig())

    def forward(
        self,
        cue_packet: CuePacket,
        active_region: ActiveRegion,
        basin_state: BasinState,
        *,
        steps: int | None = None,
    ) -> FieldLoopOutput:
        """Run context → binding → forces → interference → settling for K steps."""

        steps = steps if steps is not None else self.config.settling_steps
        batch_size = active_region.trace_amp.shape[0]
        basin_state = self._ensure_basin_state(basin_state, batch_size=batch_size)
        trace_amp = active_region.trace_amp
        basin_pressures = basin_state.pressures

        binding_state: BindingState | None = None
        interference_state: InterferenceState | None = None

        for step in range(steps):
            step_region = replace(active_region, trace_amp=trace_amp)
            context = self.context_op(cue_packet, step_region)
            binding_state = self.binding_layer(step_region, basin_state, context)
            forces = self.binding_forces(step_region, basin_state, binding_state, context)
            step_basin_state = replace(basin_state, pressures=basin_pressures)
            interference_state = self.interference_layer(step_region, binding_state, step_basin_state)

            interference_drive = self._interference_trace_drive(
                interference_state,
                num_traces=trace_amp.shape[-1],
            )
            total_trace_force = (
                forces.trace_force
                + context.trace_drive
                + interference_drive
                - trace_amp * 0.01
            )
            trace_amp = self.settling(trace_amp, total_trace_force)

            basin_interference = compose_basin_interference_force(interference_state)
            basin_force = forces.basin_force + context.basin_drive + basin_interference
            basin_pressures = self.settling(basin_pressures, basin_force)

            _log_trace(
                self.logger,
                self.config.trace,
                "field_loop.step",
                step=step + 1,
                trace_amp_mean=float(trace_amp.mean().item()),
                basin_mean=float(basin_pressures.mean().item()),
                conflict_score=float(interference_state.conflict_score.mean().item())
                if interference_state.conflict_score is not None
                else 0.0,
                basin_interference_mean=float(basin_interference.mean().item()),
            )

        if binding_state is None or interference_state is None:
            raise RuntimeError("field loop did not run any settling steps.")

        return FieldLoopOutput(
            active_region_trace_amp=trace_amp,
            basin_pressures=basin_pressures,
            binding_state=binding_state,
            interference_state=interference_state,
            steps_run=steps,
        )

    @staticmethod
    def _interference_trace_drive(
        interference_state: InterferenceState,
        *,
        num_traces: int,
    ) -> Tensor:
        if interference_state.pair_energy is None:
            raise ValueError("interference_state.pair_energy is required.")
        return interference_state.pair_energy.expand(-1, num_traces)

    def _ensure_basin_state(self, basin_state: BasinState, *, batch_size: int) -> BasinState:
        spec = BasinStateSpec(
            batch_size=batch_size,
            num_basins=self.config.num_basins,
            basin_dim=self.config.content_dim,
        )
        validate_basin_state(basin_state, spec=spec)
        if basin_state.vectors.data_ptr() != self.basin_bank.vectors.data_ptr():
            if not torch.allclose(basin_state.vectors, self.basin_bank.vectors):
                raise ValueError(
                    "basin_state.vectors must match field_loop.basin_bank.vectors; "
                    "use basin_bank.batch_state() to construct basin state."
                )
        return basin_state

    def make_basin_state(self, batch_size: int, *, device: torch.device | None = None) -> BasinState:
        """Create a batch basin state backed by this loop's learnable basin bank."""

        return self.basin_bank.batch_state(batch_size, device=device)


def run_field_loop_on_text(
    text: str,
    *,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    num_basins: int = 32,
    settling_steps: int = 3,
    seed: int = 7,
    trace: bool = False,
) -> FieldLoopOutput:
    """Route text into an active region, then run the field loop."""

    cue_packet, _routing, active_region = route_text(
        text,
        num_traces=num_traces,
        top_k=top_k,
        cue_dim=cue_dim,
        seed=seed,
        trace=trace,
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
    basin_state = field_loop.make_basin_state(
        active_region.trace_amp.shape[0],
        device=active_region.trace_amp.device,
    )
    field_loop.eval()
    with torch.no_grad():
        return field_loop(cue_packet, active_region, basin_state)


def _configure_logging(*, trace: bool) -> None:
    if not trace:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        force=True,
    )


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse field settling on text.")
    parser.add_argument(
        "text",
        nargs="+",
        help='Text to process. You may optionally start with the word "text".',
    )
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--num-basins", type=int, default=32)
    parser.add_argument("--settling-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)

    output = run_field_loop_on_text(
        text,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        num_basins=args.num_basins,
        settling_steps=args.settling_steps,
        seed=args.seed,
        trace=args.trace,
    )

    _safe_print(f"Input: {text}")
    _safe_print(f"Steps: {output.steps_run}")
    _safe_print(f"Trace amp shape: {list(output.active_region_trace_amp.shape)}")
    _safe_print(f"Basin pressure shape: {list(output.basin_pressures.shape)}")
    field_loop = FieldLoop(
        FieldLoopConfig(
            cue_dim=args.cue_dim,
            content_dim=args.cue_dim,
            active_traces=args.top_k,
            num_basins=args.num_basins,
            settling_steps=args.settling_steps,
        )
    )
    _safe_print(f"Learnable scalars in field loop: {sum(p.numel() for p in field_loop.parameters())}")


__all__ = [
    "FieldLoop",
    "FieldLoopConfig",
    "FieldLoopOutput",
    "make_placeholder_basin_state",
    "run_field_loop_on_text",
]


if __name__ == "__main__":
    main()
