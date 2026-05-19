import subprocess
import sys

from lmf.infra.scripts.inspect_pipeline import inspect_pipeline
from lmf.core.support.human_report import format_pipeline_report


def test_format_pipeline_report_lists_active_traces_in_plain_language() -> None:
    inspection = inspect_pipeline(
        "Help me withdraw money from the bank",
        num_traces=24,
        top_k=5,
        cue_dim=8,
        settling_steps=2,
        seed=3,
    )
    report = format_pipeline_report(inspection)

    assert "ROUTING" in report
    assert "trace_" in report
    assert "match" in report
    assert "bank" in report
    assert "BINDING" in report
    assert "SETTLING" in report


def test_inspect_pipeline_cli_prints_readable_sections() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/infra/scripts/inspect_pipeline.py",
            "text",
            "Help bank!",
            "--num-traces",
            "20",
            "--top-k",
            "4",
            "--cue-dim",
            "6",
            "--settling-steps",
            "2",
            "--seed",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TOKENS" in result.stdout
    assert "ROUTING" in result.stdout
    assert "trace_" in result.stdout
    assert "tokens used:" in result.stdout


def test_inspect_pipeline_routing_only_skips_settling_section() -> None:
    inspection = inspect_pipeline(
        "Help bank!",
        num_traces=16,
        top_k=4,
        cue_dim=6,
        settling_steps=2,
        seed=2,
        run_field=False,
    )
    report = format_pipeline_report(inspection)

    assert "ROUTING" in report
    assert "SETTLING" not in report
    assert "BINDING" not in report
