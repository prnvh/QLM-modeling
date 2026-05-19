import logging
import subprocess
import sys

import pytest
import torch

from lmf.core.dmf.active_region import build_active_region
from lmf.core.dmf.sparsity import (
    SparsityReport,
    active_trace_fraction,
    resolve_top_k,
    topk_trace_routing,
)
from lmf.core.dmf.trace_bank import TraceBank, TraceBankConfig
from lmf.core.dmf.trace_router import (
    RoutingResult,
    TraceRouter,
    TraceRouterConfig,
    route_text,
)
from lmf.core.input.cue_packet import CuePacket
from lmf.core.state.types import ActiveRegion


def _make_bank_and_cues(
    *,
    num_traces: int = 16,
    cue_dim: int = 6,
    batch_size: int = 2,
    seq_len: int = 4,
) -> tuple[TraceBank, CuePacket]:
    torch.manual_seed(11)
    bank = TraceBank(
        TraceBankConfig(
            num_traces=num_traces,
            key_dim=cue_dim,
            content_dim=cue_dim,
        )
    )
    cues = torch.randn(batch_size, seq_len, cue_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    if seq_len >= 2:
        mask[:, -1] = False
    if batch_size >= 2 and seq_len >= 3:
        mask[1, 2:] = False
    pooled = cues.sum(dim=1) / mask.sum(dim=1, keepdim=True).to(dtype=cues.dtype)
    token_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1) + 3
    positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return bank, CuePacket(
        cues=cues,
        mask=mask,
        pooled=pooled,
        token_ids=token_ids,
        positions=positions,
        special_token_ids=(0, 1, 2, 3, 4),
    )


def test_topk_trace_routing_returns_expected_shapes() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.4],
            [0.3, 0.1, 0.8, 0.5],
        ]
    )
    trace_ids, top_scores = topk_trace_routing(scores, top_k=2)

    assert trace_ids.shape == (2, 2)
    assert top_scores.shape == (2, 2)
    assert trace_ids[0].tolist() == [1, 3]
    assert trace_ids[1].tolist() == [2, 3]


def test_resolve_top_k_clamps_to_bank_size() -> None:
    assert resolve_top_k(requested_k=32, num_traces=10) == 10


def test_sparsity_report_tracks_active_fraction() -> None:
    report = SparsityReport.from_routing(batch_size=2, num_traces=64, active_traces=8)

    assert report.active_traces == 8
    assert report.inactive_traces == 56
    assert report.active_fraction == active_trace_fraction(active_traces=8, num_traces=64)


def test_trace_router_returns_top_k_ids_not_full_bank() -> None:
    bank, cue_packet = _make_bank_and_cues(num_traces=20, cue_dim=5)
    router = TraceRouter(TraceRouterConfig(top_k=4, routing_mode="max_token"))

    routing = router(cue_packet, bank)

    assert isinstance(routing, RoutingResult)
    assert routing.trace_ids.shape == (2, 4)
    assert routing.scores.shape == (2, 4)
    assert routing.sparsity.active_traces == 4
    assert routing.sparsity.num_traces == 20
    assert routing.sparsity.active_fraction == 0.2


def test_trace_router_is_deterministic_with_seed() -> None:
    bank_a, cue_packet_a = _make_bank_and_cues()
    bank_b, cue_packet_b = _make_bank_and_cues()
    router = TraceRouter(TraceRouterConfig(top_k=3))

    first = router(cue_packet_a, bank_a)
    second = router(cue_packet_b, bank_b)

    assert torch.equal(first.trace_ids, second.trace_ids)
    assert torch.allclose(first.scores, second.scores)


def test_trace_router_pooled_mode_uses_packet_pool() -> None:
    bank, cue_packet = _make_bank_and_cues(batch_size=1, seq_len=3)
    router = TraceRouter(TraceRouterConfig(top_k=2, routing_mode="pooled"))
    pooled_routing = router(cue_packet, bank)

    manual_pool = cue_packet.pooled
    assert manual_pool is not None
    scores = manual_pool @ bank.state().keys.transpose(0, 1)
    scores = scores - bank.state().threshold.unsqueeze(0)
    expected_ids, _ = topk_trace_routing(scores, top_k=2)

    assert torch.equal(pooled_routing.trace_ids, expected_ids)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TraceRouterConfig(top_k=0),
        lambda: TraceRouterConfig(top_k=4, routing_mode="bad"),  # type: ignore[arg-type]
        lambda: TraceRouterConfig(top_k=4, score_mode="bad"),  # type: ignore[arg-type]
    ],
)
def test_trace_router_config_rejects_invalid_values(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_trace_router_rejects_mismatched_cue_width() -> None:
    bank = TraceBank(TraceBankConfig(num_traces=8, key_dim=4, content_dim=4))
    packet = CuePacket(
        cues=torch.randn(1, 3, 6),
        token_ids=torch.tensor([[1, 2, 3]]),
    )
    router = TraceRouter(TraceRouterConfig(top_k=2))

    with pytest.raises(ValueError, match="cue width"):
        router(packet, bank)


def test_trace_router_rejects_missing_token_ids() -> None:
    bank = TraceBank(TraceBankConfig(num_traces=8, key_dim=4, content_dim=4))
    packet = CuePacket(cues=torch.randn(1, 3, 4))
    router = TraceRouter(TraceRouterConfig(top_k=2))

    with pytest.raises(ValueError, match="token_ids"):
        router(packet, bank)


def test_trace_router_trace_logs_human_readable_routing(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lmf.core.dmf.trace_router")
    bank, cue_packet = _make_bank_and_cues(batch_size=1)
    router = TraceRouter(TraceRouterConfig(top_k=3, trace=True, trace_limit=2))

    router(cue_packet, bank)

    messages = [record.message for record in caplog.records]
    assert any("trace_router.forward" in message for message in messages)
    assert any("top_k=3" in message for message in messages)


def test_build_active_region_materializes_sparse_packet() -> None:
    bank, cue_packet = _make_bank_and_cues(num_traces=12, cue_dim=4)
    router = TraceRouter(TraceRouterConfig(top_k=3))
    routing = router(cue_packet, bank)

    active_region = build_active_region(
        trace_bank=bank,
        routing=routing,
        cue_packet=cue_packet,
    )

    assert isinstance(active_region, ActiveRegion)
    assert active_region.trace_ids.shape == (2, 3)
    assert active_region.trace_content.shape == (2, 3, 4)
    assert active_region.trace_amp.shape == (2, 3)
    assert active_region.cue_drive.shape == (2, 3)
    assert active_region.mask is not None
    assert active_region.mask.dtype == torch.bool

    flat_ids = routing.trace_ids.reshape(-1)
    expected_content = bank.content.index_select(0, flat_ids).reshape(2, 3, 4)
    assert torch.allclose(active_region.trace_content, expected_content)


def test_trace_router_attaches_provenance_for_max_token_mode() -> None:
    bank, cue_packet = _make_bank_and_cues(num_traces=12, cue_dim=4, batch_size=1, seq_len=3)
    router = TraceRouter(TraceRouterConfig(top_k=2, routing_mode="max_token"))
    routing = router(cue_packet, bank)

    assert routing.source_cue_id.shape == (1, 2)
    assert routing.source_token_id.shape == (1, 2)
    assert routing.source_span.shape == (1, 2, 2)
    assert routing.cue_type.shape == (1, 2)
    assert routing.normalized_cue_ids.shape == (1, 2)
    assert (routing.source_cue_id >= 0).all()
    assert (routing.source_token_id >= 0).all()


def test_trace_router_pooled_provenance_marks_sentence_level_cues() -> None:
    bank, cue_packet = _make_bank_and_cues(num_traces=12, cue_dim=4, batch_size=1, seq_len=4)
    router = TraceRouter(TraceRouterConfig(top_k=2, routing_mode="pooled"))
    routing = router(cue_packet, bank)

    assert (routing.source_cue_id == -1).all()
    assert (routing.source_token_id == -1).all()
    assert (routing.cue_type == 6).all()  # CUE_TYPE_POOLED
    assert (routing.normalized_cue_ids == -1).all()
    assert routing.source_span[0, 0, 0].item() <= routing.source_span[0, 0, 1].item()


def test_build_active_region_copies_provenance_from_routing() -> None:
    bank, cue_packet = _make_bank_and_cues(num_traces=12, cue_dim=4, batch_size=1)
    router = TraceRouter(TraceRouterConfig(top_k=3))
    routing = router(cue_packet, bank)
    active_region = build_active_region(
        trace_bank=bank,
        routing=routing,
        cue_packet=cue_packet,
    )

    assert torch.equal(active_region.source_cue_id, routing.source_cue_id)
    assert torch.equal(active_region.source_token_id, routing.source_token_id)
    assert torch.equal(active_region.source_span, routing.source_span)
    assert torch.equal(active_region.cue_type, routing.cue_type)
    assert torch.equal(active_region.normalized_cue_ids, routing.normalized_cue_ids)


def test_route_text_active_region_carries_provenance() -> None:
    _cue_packet, routing, active_region = route_text(
        "Help bank!",
        num_traces=24,
        top_k=5,
        cue_dim=6,
        seed=13,
    )

    assert active_region.source_cue_id is not None
    assert active_region.normalized_cue_ids is not None
    assert active_region.source_cue_id.shape == routing.trace_ids.shape
    assert int(active_region.source_cue_id.min().item()) >= 0


def test_trace_router_provenance_matches_winning_token_cue() -> None:
    torch.manual_seed(0)
    bank = TraceBank(TraceBankConfig(num_traces=6, key_dim=3, content_dim=3))
    bank.keys.data.zero_()
    bank.threshold.data.fill_(-10.0)

    cues = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    keys = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    bank.keys.data.copy_(keys)
    token_ids = torch.tensor([[3, 7, 11]])
    packet = CuePacket(
        cues=cues,
        mask=torch.ones(1, 3, dtype=torch.bool),
        token_ids=token_ids,
        positions=torch.tensor([[0, 1, 2]]),
        special_token_ids=(0, 1, 2, 3, 4),
    )
    router = TraceRouter(TraceRouterConfig(top_k=3, routing_mode="max_token", score_mode="dot"))
    routing = router(packet, bank)

    # Trace 1 is driven by cue 0, trace 2 by cue 1, trace 3 by cue 2.
    ranked = sorted(
        zip(
            routing.trace_ids[0].tolist(),
            routing.source_cue_id[0].tolist(),
            routing.source_token_id[0].tolist(),
        ),
        key=lambda row: row[0],
    )
    assert ranked == [(1, 0, 3), (2, 1, 7), (3, 2, 11)]


def test_route_text_builds_viewable_active_region() -> None:
    _cue_packet, routing, active_region = route_text(
        "Help bank!",
        num_traces=24,
        top_k=5,
        cue_dim=6,
        seed=13,
    )

    assert routing.trace_ids.shape == (1, 5)
    assert active_region.trace_content.shape == (1, 5, 6)


def test_trace_router_file_can_be_run_directly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/core/dmf/trace_router.py",
            "text",
            "Help bank!",
            "--num-traces",
            "24",
            "--top-k",
            "5",
            "--cue-dim",
            "6",
            "--seed",
            "13",
            "--trace",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ROUTING" in result.stdout
    assert "trace_" in result.stdout
    assert "from token" in result.stdout
    assert "trace_router.forward" in result.stderr
