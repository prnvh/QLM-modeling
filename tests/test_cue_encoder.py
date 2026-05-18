import logging
import subprocess
import sys

import pytest
import torch
from torch import nn

from lmf.core.input.cue_encoder import CueEncoder, CueEncoderConfig, encode_text
from lmf.core.input.cue_packet import CuePacket


def test_cue_encoder_outputs_packet_shapes_and_masks_padding() -> None:
    torch.manual_seed(3)
    encoder = CueEncoder(
        CueEncoderConfig(
            vocab_size=10,
            cue_dim=6,
            embedding_dim=5,
            hidden_dim=7,
            pad_id=0,
            window_size=3,
        )
    )
    input_ids = torch.tensor(
        [
            [3, 5, 6, 4, 0],
            [3, 5, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    packet = encoder(input_ids)

    assert isinstance(packet, CuePacket)
    assert packet.cues.shape == (2, 5, 6)
    assert packet.mask.tolist() == [
        [True, True, True, True, False],
        [True, True, False, False, False],
    ]
    assert packet.positions.tolist() == [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]
    assert packet.pooled.shape == (2, 6)
    assert torch.isfinite(packet.cues).all()
    assert torch.isfinite(packet.pooled).all()
    assert torch.allclose(packet.cues[0, 4], torch.zeros(6), atol=1e-6)
    assert torch.allclose(packet.cues[1, 2:], torch.zeros(3, 6), atol=1e-6)

    manual_pooled = packet.cues.sum(dim=1) / packet.mask.sum(dim=1, keepdim=True)
    assert torch.allclose(packet.pooled, manual_pooled, atol=1e-6)


def test_cue_encoder_accepts_single_sequence_and_custom_mask_positions() -> None:
    torch.manual_seed(5)
    encoder = CueEncoder(CueEncoderConfig(vocab_size=12, cue_dim=4, pad_id=0))
    input_ids = torch.tensor([3, 7, 8, 4], dtype=torch.long)
    mask = torch.tensor([1, 1, 0, 1], dtype=torch.long)
    positions = torch.tensor([10, 11, 12, 13], dtype=torch.long)

    packet = encoder(input_ids, attention_mask=mask, positions=positions)

    assert packet.cues.shape == (1, 4, 4)
    assert packet.mask.tolist() == [[True, True, False, True]]
    assert packet.positions.tolist() == [[10, 11, 12, 13]]
    assert torch.allclose(packet.cues[0, 2], torch.zeros(4), atol=1e-6)


def test_cue_encoder_has_no_transformer_attention_modules() -> None:
    encoder = CueEncoder(CueEncoderConfig(vocab_size=10, cue_dim=4))

    forbidden = (nn.MultiheadAttention, nn.TransformerEncoder, nn.TransformerDecoder)
    assert not any(isinstance(module, forbidden) for module in encoder.modules())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CueEncoderConfig(vocab_size=0, cue_dim=4),
        lambda: CueEncoderConfig(vocab_size=4, cue_dim=0),
        lambda: CueEncoderConfig(vocab_size=4, cue_dim=4, pad_id=4),
        lambda: CueEncoderConfig(vocab_size=4, cue_dim=4, window_size=2),
        lambda: CueEncoderConfig(vocab_size=4, cue_dim=4, dropout=1.0),
        lambda: CueEncoderConfig(vocab_size=4, cue_dim=4, trace_limit=-1),
    ],
)
def test_cue_encoder_config_rejects_brittle_arguments(factory) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "input_ids,kwargs",
    [
        (torch.empty(0, dtype=torch.long), {}),
        (torch.zeros(1, 2, 3, dtype=torch.long), {}),
        (torch.tensor([1.5, 2.5]), {}),
        (torch.tensor([1, 99], dtype=torch.long), {}),
        (torch.tensor([[1, 2]], dtype=torch.long), {"attention_mask": torch.tensor([[1, 1, 1]])}),
        (torch.tensor([[1, 2]], dtype=torch.long), {"positions": torch.tensor([[0, 1, 2]])}),
    ],
)
def test_cue_encoder_forward_rejects_invalid_inputs(input_ids, kwargs) -> None:
    encoder = CueEncoder(CueEncoderConfig(vocab_size=8, cue_dim=4))

    with pytest.raises(ValueError):
        encoder(input_ids, **kwargs)


def test_cue_encoder_trace_logs_human_readable_packet(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lmf.core.input.cue_encoder")
    encoder = CueEncoder(CueEncoderConfig(vocab_size=10, cue_dim=4, trace=True, trace_limit=3))

    encoder(torch.tensor([[3, 5, 4, 0]], dtype=torch.long))

    messages = [record.message for record in caplog.records]
    assert any("cue_encoder.forward" in message for message in messages)
    assert any("input_shape=[1, 4]" in message for message in messages)
    assert any("cue_shape=[1, 4, 4]" in message for message in messages)
    assert any("active_tokens=3" in message for message in messages)


def test_encode_text_builds_viewable_cue_packet() -> None:
    vocab, ids, packet = encode_text("Help bank!", cue_dim=5, seed=11)

    assert len(vocab) == 8
    assert ids[0] == vocab.bos_id
    assert ids[-1] == vocab.eos_id
    assert packet.cues.shape == (1, 5, 5)
    assert packet.mask.tolist() == [[True, True, True, True, True]]


def test_cue_encoder_file_can_be_run_directly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "lmf/core/input/cue_encoder.py",
            "text",
            "Help bank!",
            "--cue-dim",
            "6",
            "--embedding-dim",
            "5",
            "--hidden-dim",
            "7",
            "--seed",
            "13",
            "--trace",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Input: Help bank!" in result.stdout
    assert "Vocab size: 8" in result.stdout
    assert "Token IDs: [3, 7, 6, 5, 4]" in result.stdout
    assert "Cue shape: [1, 5, 6]" in result.stdout
    assert "Pooled cue:" in result.stdout
    assert "cue_encoder.forward" in result.stderr
