"""Commit B2: field submodules must be registered and checkpointed."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from lmf.core.field.loop import FieldLoop, FieldLoopConfig

B2_PARAMETER_PREFIXES = (
    "binding_layer.",
    "binding_forces.",
    "interference_layer.",
)

B2_STATE_DICT_PREFIXES = B2_PARAMETER_PREFIXES


def _build_field_loop() -> FieldLoop:
    return FieldLoop(
        FieldLoopConfig(
            cue_dim=8,
            content_dim=8,
            active_traces=6,
            num_basins=12,
            relation_channels=4,
            settling_steps=2,
        )
    )


def _prefixes_present(keys: list[str], required_prefixes: tuple[str, ...]) -> list[str]:
    missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in keys)]
    return missing


@pytest.mark.parametrize("prefix", B2_PARAMETER_PREFIXES)
def test_b2_required_submodules_appear_in_named_parameters(prefix: str) -> None:
    field_loop = _build_field_loop()
    parameter_names = [name for name, _ in field_loop.named_parameters()]

    assert any(name.startswith(prefix) for name in parameter_names), (
        f"expected named_parameters key starting with {prefix!r}, got {parameter_names}"
    )


def test_b2_named_parameters_include_all_required_submodules() -> None:
    field_loop = _build_field_loop()
    parameter_names = [name for name, _ in field_loop.named_parameters()]
    missing = _prefixes_present(parameter_names, B2_PARAMETER_PREFIXES)

    assert not missing, f"missing named_parameters prefixes: {missing}"


@pytest.mark.parametrize("prefix", B2_STATE_DICT_PREFIXES)
def test_b2_required_submodules_appear_in_state_dict(prefix: str) -> None:
    field_loop = _build_field_loop()
    state_keys = list(field_loop.state_dict().keys())

    assert any(key.startswith(prefix) for key in state_keys), (
        f"expected state_dict key starting with {prefix!r}, got {state_keys}"
    )


def test_b2_state_dict_matches_named_parameters() -> None:
    field_loop = _build_field_loop()
    parameter_names = {name for name, _ in field_loop.named_parameters()}
    state_keys = set(field_loop.state_dict().keys())

    assert parameter_names == state_keys


def test_b2_each_required_submodule_has_trainable_parameters() -> None:
    field_loop = _build_field_loop()
    grouped = {prefix: [] for prefix in B2_PARAMETER_PREFIXES}

    for name, parameter in field_loop.named_parameters():
        for prefix in B2_PARAMETER_PREFIXES:
            if name.startswith(prefix):
                grouped[prefix].append(parameter)

    for prefix, parameters in grouped.items():
        assert parameters, f"no parameters found for {prefix}"
        assert sum(param.numel() for param in parameters) > 0


def test_b2_checkpoint_buffer_roundtrip_preserves_tensors() -> None:
    torch.manual_seed(21)
    field_loop = _build_field_loop()
    before = field_loop.state_dict()

    buffer = io.BytesIO()
    torch.save(before, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)

    reloaded = _build_field_loop()
    reloaded.load_state_dict(restored)

    after = reloaded.state_dict()
    assert set(before.keys()) == set(after.keys())
    for key in B2_STATE_DICT_PREFIXES:
        assert any(name.startswith(key) for name in after), f"checkpoint missing {key}"

    for key, tensor in before.items():
        assert torch.equal(tensor, after[key]), f"checkpoint mismatch for {key}"


def test_b2_field_loop_inside_parent_model_is_checkpointed() -> None:
    """Guard against field_loop living outside the module tree."""

    class Stage1Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.field_loop = _build_field_loop()

    torch.manual_seed(4)
    model = Stage1Model()
    before = model.state_dict()

    buffer = io.BytesIO()
    torch.save(before, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)

    clone = Stage1Model()
    clone.load_state_dict(restored)

    for prefix in B2_PARAMETER_PREFIXES:
        parent_key = f"field_loop.{prefix}"
        assert any(key.startswith(parent_key) for key in before), parent_key
        assert any(key.startswith(parent_key) for key in clone.state_dict()), parent_key

    for key, tensor in before.items():
        if key.startswith("field_loop."):
            assert torch.equal(tensor, clone.state_dict()[key])


def test_b2_unregistered_module_would_fail_parameter_scan() -> None:
    """Regression guard: detached nn.Module children must not be the only storage."""

    field_loop = _build_field_loop()
    detached = field_loop.binding_layer
    assert "binding_layer" in dict(field_loop.named_children())

    # If binding_layer were not registered, it would not appear under field_loop.* keys.
    assert any(name.startswith("binding_layer.") for name, _ in field_loop.named_parameters())
    assert id(detached) == id(field_loop.binding_layer)
