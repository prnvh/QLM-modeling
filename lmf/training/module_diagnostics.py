"""Training-time parameter counts and gradient norms for major LMF modules."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.dmf.trace_bank import TraceBank, TraceBankConfig  # noqa: E402
from lmf.core.dmf.trace_router import route_text  # noqa: E402
from lmf.core.field.loop import FieldLoop, FieldLoopConfig  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleDiagnosticReport:
    """B3 metrics: gradient norms and parameter counts for core modules."""

    trace_bank_grad_norm: float | None
    binding_layer_grad_norm: float | None
    binding_forces_grad_norm: float | None
    interference_grad_norm: float | None
    basin_grad_norm: float | None
    decoder_grad_norm: float | None
    field_loop_param_count: int
    binding_param_count: int
    interference_param_count: int

    def to_log_fields(self) -> dict[str, str]:
        """Human-readable key=value fields for logging."""

        fields: dict[str, str] = {}
        for key, value in asdict(self).items():
            if value is None:
                fields[key] = "n/a"
            elif isinstance(value, float):
                fields[key] = f"{value:.6f}"
            else:
                fields[key] = str(value)
        return fields


def parameter_count(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters())


def gradient_norm(module: nn.Module | None) -> float | None:
    """Return L2 norm of all gradients on a module, or None if no grads exist."""

    if module is None:
        return None

    squared_sum = 0.0
    found = False
    for param in module.parameters():
        if param.grad is None:
            continue
        found = True
        squared_sum += float(param.grad.detach().pow(2).sum().item())

    if not found:
        return None
    return math.sqrt(squared_sum)


def resolve_submodule(model: nn.Module, *paths: str) -> nn.Module | None:
    """Return the first submodule found at a dotted attribute path."""

    for path in paths:
        current: Any = model
        for part in path.split("."):
            if not hasattr(current, part):
                current = None
                break
            current = getattr(current, part)
        if isinstance(current, nn.Module):
            return current
    return None


def collect_module_diagnostics(model: nn.Module) -> ModuleDiagnosticReport:
    """Collect B3 gradient norms and parameter counts from a model tree."""

    trace_bank = resolve_submodule(model, "trace_bank")
    field_loop = resolve_submodule(model, "field_loop")
    binding_layer = resolve_submodule(model, "field_loop.binding_layer", "binding_layer")
    binding_forces = resolve_submodule(model, "field_loop.binding_forces", "binding_forces")
    interference_layer = resolve_submodule(
        model,
        "field_loop.interference_layer",
        "interference_layer",
    )
    basin_module = resolve_submodule(
        model,
        "basin_bank",
        "basin_state",
        "field_loop.basin_bank",
    )
    decoder = resolve_submodule(model, "decoder")

    binding_modules = [module for module in (binding_layer, binding_forces) if module is not None]

    return ModuleDiagnosticReport(
        trace_bank_grad_norm=gradient_norm(trace_bank),
        binding_layer_grad_norm=gradient_norm(binding_layer),
        binding_forces_grad_norm=gradient_norm(binding_forces),
        interference_grad_norm=gradient_norm(interference_layer),
        basin_grad_norm=gradient_norm(basin_module),
        decoder_grad_norm=gradient_norm(decoder),
        field_loop_param_count=parameter_count(field_loop),
        binding_param_count=sum(parameter_count(module) for module in binding_modules),
        interference_param_count=parameter_count(interference_layer),
    )


def format_module_diagnostics(report: ModuleDiagnosticReport) -> str:
    """Single-line human-readable summary."""

    return " | ".join(f"{key}={value}" for key, value in report.to_log_fields().items())


def log_module_diagnostics(
    model: nn.Module,
    *,
    logger: logging.Logger = LOGGER,
    step: int | None = None,
    event: str = "module_diagnostics",
) -> ModuleDiagnosticReport:
    """Log B3 diagnostics at INFO level."""

    report = collect_module_diagnostics(model)
    fields = report.to_log_fields()
    if step is not None:
        fields = {"step": str(step), **fields}
    details = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s | %s", event, details)
    return report


class Stage1DiagnosticModel(nn.Module):
    """Minimal trainable stack used to smoke-test module diagnostics."""

    def __init__(
        self,
        *,
        num_traces: int,
        top_k: int,
        cue_dim: int,
        num_basins: int,
        settling_steps: int = 2,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.cue_dim = cue_dim
        self.num_basins = num_basins
        self.trace_bank = TraceBank(
            TraceBankConfig(
                num_traces=num_traces,
                key_dim=cue_dim,
                content_dim=cue_dim,
            )
        )
        self.field_loop = FieldLoop(
            FieldLoopConfig(
                cue_dim=cue_dim,
                content_dim=cue_dim,
                active_traces=top_k,
                num_basins=num_basins,
                settling_steps=settling_steps,
            )
        )
        self.decoder = nn.Linear(cue_dim, cue_dim)

    def forward_from_routed(
        self,
        cue_packet,
        active_region,
    ):
        basin_state = self.field_loop.make_basin_state(
            active_region.trace_amp.shape[0],
            device=active_region.trace_amp.device,
        )
        field_output = self.field_loop(cue_packet, active_region, basin_state)
        pooled = cue_packet.pooled
        if pooled is None:
            pooled = cue_packet.cues.mean(dim=1)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        decoded = self.decoder(pooled)
        return field_output, decoded


def run_diagnostics_demo(
    text: str,
    *,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    num_basins: int = 32,
    settling_steps: int = 2,
    seed: int = 7,
) -> tuple[ModuleDiagnosticReport, Stage1DiagnosticModel]:
    """Forward + backward on a small stack, then return diagnostics and the model."""

    torch.manual_seed(seed)
    cue_packet, _routing, active_region = route_text(
        text,
        num_traces=num_traces,
        top_k=top_k,
        cue_dim=cue_dim,
        seed=seed,
    )
    model = Stage1DiagnosticModel(
        num_traces=num_traces,
        top_k=top_k,
        cue_dim=cue_dim,
        num_basins=num_basins,
        settling_steps=settling_steps,
    )
    model.train()
    field_output, decoded = model.forward_from_routed(cue_packet, active_region)
    loss = field_output.active_region_trace_amp.sum() + field_output.basin_pressures.sum() + decoded.sum()
    loss.backward()
    return collect_module_diagnostics(model), model


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
    parser = argparse.ArgumentParser(
        description="Run one training step and print module gradient norms / parameter counts.",
    )
    parser.add_argument(
        "text",
        nargs="+",
        help='Text to process. You may optionally start with the word "text".',
    )
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--num-basins", type=int, default=32)
    parser.add_argument("--settling-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true", help="Also log diagnostics through the logger.")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _configure_logging(trace=args.trace)
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)

    report, model = run_diagnostics_demo(
        text,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        num_basins=args.num_basins,
        settling_steps=args.settling_steps,
        seed=args.seed,
    )

    if args.trace:
        log_module_diagnostics(model, step=0)

    _safe_print(f"Input: {text}")
    _safe_print(format_module_diagnostics(report))


__all__ = [
    "ModuleDiagnosticReport",
    "Stage1DiagnosticModel",
    "collect_module_diagnostics",
    "format_module_diagnostics",
    "gradient_norm",
    "log_module_diagnostics",
    "parameter_count",
    "resolve_submodule",
    "run_diagnostics_demo",
]


if __name__ == "__main__":
    main()
