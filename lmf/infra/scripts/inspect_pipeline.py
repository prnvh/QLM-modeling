"""Readable end-to-end inspection: text → routing → binding → settling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lmf.core.dmf.trace_router import route_text  # noqa: E402
from lmf.core.field.loop import (  # noqa: E402
    FieldLoop,
    FieldLoopConfig,
    make_placeholder_basin_state,
)
from lmf.core.input.tokenizer import inspect_text  # noqa: E402
from lmf.core.support.human_report import (  # noqa: E402
    PipelineInspection,
    format_pipeline_report,
    resolve_source_tokens,
    tensors_to_ranked_lists,
)


def inspect_pipeline(
    text: str,
    *,
    num_traces: int = 64,
    top_k: int = 8,
    cue_dim: int = 16,
    num_basins: int = 32,
    settling_steps: int = 3,
    routing_mode: str = "max_token",
    score_mode: str = "dot",
    seed: int = 7,
    run_field: bool = True,
) -> PipelineInspection:
    token_report = inspect_text(text)
    cue_packet, routing, active_region = route_text(
        text,
        num_traces=num_traces,
        top_k=top_k,
        cue_dim=cue_dim,
        routing_mode=routing_mode,  # type: ignore[arg-type]
        score_mode=score_mode,  # type: ignore[arg-type]
        seed=seed,
        trace=False,
    )

    trace_ids, match_scores, activations_before, cue_drives, source_cue_ids, source_token_ids, source_spans, cue_types, normalized = (
        tensors_to_ranked_lists(
            routing.trace_ids,
            routing.scores,
            active_region.trace_amp,
            active_region.cue_drive,
            source_cue_id=routing.source_cue_id,
            source_token_id=routing.source_token_id,
            source_span=routing.source_span,
            cue_type=routing.cue_type,
            normalized_cue_ids=routing.normalized_cue_ids,
        )
    )

    field_output = None
    activations_after = None
    if run_field:
        basin_state = make_placeholder_basin_state(
            batch_size=active_region.trace_amp.shape[0],
            num_basins=num_basins,
            basin_dim=cue_dim,
            device=active_region.trace_amp.device,
            dtype=active_region.trace_amp.dtype,
        )
        field_loop = FieldLoop(
            FieldLoopConfig(
                cue_dim=cue_dim,
                content_dim=cue_dim,
                active_traces=top_k,
                num_basins=num_basins,
                settling_steps=settling_steps,
            )
        )
        field_loop.eval()
        with torch.no_grad():
            field_output = field_loop(cue_packet, active_region, basin_state)

        after_by_id = {
            int(routing.trace_ids[0, index].item()): float(
                field_output.active_region_trace_amp[0, index].item()
            )
            for index in range(routing.trace_ids.shape[1])
        }
        activations_after = [after_by_id[trace_id] for trace_id in trace_ids]

    source_tokens = resolve_source_tokens(
        list(token_report["token_ids"]),  # type: ignore[arg-type]
        source_cue_ids,
        cue_types,
    )

    return PipelineInspection(
        input_text=text,
        tokens=list(token_report["tokens"]),  # type: ignore[arg-type]
        token_rows=list(token_report["token_ids"]),  # type: ignore[arg-type]
        routing_mode=routing_mode,
        score_mode=score_mode,
        num_traces=num_traces,
        top_k=top_k,
        trace_ids=trace_ids,
        match_scores=match_scores,
        activations_before=activations_before,
        cue_drives=cue_drives,
        source_cue_ids=source_cue_ids,
        source_token_ids=source_token_ids,
        source_tokens=source_tokens,
        source_spans=source_spans,
        cue_types=cue_types,
        normalized_cue_ids=normalized,
        field_output=field_output,
        activations_after=activations_after,
    )


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Human-readable report for text through sparse routing and field loop.",
    )
    parser.add_argument(
        "text",
        nargs="+",
        help='Text to inspect. You may optionally start with the word "text".',
    )
    parser.add_argument("--num-traces", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cue-dim", type=int, default=16)
    parser.add_argument("--num-basins", type=int, default=32)
    parser.add_argument("--settling-steps", type=int, default=3)
    parser.add_argument("--routing-mode", choices=("max_token", "pooled"), default="max_token")
    parser.add_argument("--score-mode", choices=("dot", "cosine"), default="dot")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="Stop after sparse routing (skip binding/settling section).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    parts = args.text[1:] if args.text and args.text[0].lower() == "text" else args.text
    text = " ".join(parts)

    inspection = inspect_pipeline(
        text,
        num_traces=args.num_traces,
        top_k=args.top_k,
        cue_dim=args.cue_dim,
        num_basins=args.num_basins,
        settling_steps=args.settling_steps,
        routing_mode=args.routing_mode,
        score_mode=args.score_mode,
        seed=args.seed,
        run_field=not args.routing_only,
    )

    encoding = sys.stdout.encoding or "utf-8"
    report = format_pipeline_report(inspection)
    print(report.encode(encoding, errors="backslashreplace").decode(encoding))


if __name__ == "__main__":
    main()
