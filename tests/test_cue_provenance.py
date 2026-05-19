import pytest
import torch

from lmf.core.dmf.cue_provenance import (
    CUE_TYPE_BOS,
    CUE_TYPE_EOS,
    CUE_TYPE_PAD,
    CUE_TYPE_POOLED,
    CUE_TYPE_TOKEN,
    SpecialTokenIds,
    build_trace_provenance,
    classify_cue_types,
    cue_span_for_position,
    normalize_cue_ids,
)


def test_classify_cue_types_marks_special_tokens() -> None:
    special = SpecialTokenIds(pad_id=0, unk_id=1, mask_id=2, bos_id=3, eos_id=4)
    token_ids = torch.tensor([[3, 7, 4, 0]])

    cue_types = classify_cue_types(token_ids, special)

    assert cue_types[0].tolist() == [CUE_TYPE_BOS, CUE_TYPE_TOKEN, CUE_TYPE_EOS, CUE_TYPE_PAD]


def test_normalize_cue_ids_preserves_vocabulary_ids() -> None:
    token_ids = torch.tensor([[3, 7, 4]])
    cue_types = torch.tensor([[CUE_TYPE_BOS, CUE_TYPE_TOKEN, 4]])

    normalized = normalize_cue_ids(token_ids, cue_types)

    assert torch.equal(normalized, token_ids)


def test_cue_span_for_position_uses_inclusive_token_bounds() -> None:
    positions = torch.tensor([[0, 1, 2, 3]])
    cue_pos = torch.tensor([[1, 3]])
    mask = torch.tensor([[True, True, False, True]])

    spans = cue_span_for_position(positions, cue_pos=cue_pos, mask=mask)

    assert spans.tolist() == [[[1, 1], [3, 3]]]


def test_build_trace_provenance_for_pooled_mode_uses_sentence_span() -> None:
    trace_ids = torch.tensor([[2, 5]])
    token_ids = torch.tensor([[3, 7, 11, 4]])
    positions = torch.tensor([[0, 1, 2, 3]])
    mask = torch.tensor([[True, True, True, True]])
    cue_types = classify_cue_types(
        token_ids,
        SpecialTokenIds(pad_id=0, unk_id=1, mask_id=2, bos_id=3, eos_id=4),
    )

    provenance = build_trace_provenance(
        trace_ids=trace_ids,
        source_cue_ids_per_trace=trace_ids,
        token_ids=token_ids,
        positions=positions,
        mask=mask,
        cue_types_per_position=cue_types,
        routing_mode="pooled",
    )

    assert provenance.source_cue_id.tolist() == [[-1, -1]]
    assert provenance.source_token_id.tolist() == [[-1, -1]]
    assert provenance.cue_type.tolist() == [[CUE_TYPE_POOLED, CUE_TYPE_POOLED]]
    assert provenance.source_span.tolist() == [[[0, 3], [0, 3]]]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SpecialTokenIds(pad_id=-1, unk_id=1, mask_id=2, bos_id=3, eos_id=4),
    ],
)
def test_special_token_ids_reject_negative_values(factory) -> None:
    with pytest.raises(ValueError):
        factory()
