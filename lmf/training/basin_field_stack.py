"""End-to-end field path for basin-family contrastive training (Commit D3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import torch
from torch import Tensor, nn

from lmf.core.basin.basin_family_contrastive import (
    BasinFamilyContrastive,
    BasinFamilyContrastiveConfig,
    BasinFamilyContrastiveMetrics,
    compute_basin_attractor_state,
    family_indices_tensor,
)
from lmf.core.dmf.trace_bank import TraceBank, TraceBankConfig
from lmf.core.dmf.trace_router import TraceRouter, TraceRouterConfig
from lmf.core.field.loop import FieldLoop, FieldLoopConfig
from lmf.core.input.cue_encoder import CueEncoder, CueEncoderConfig
from lmf.core.input.cue_packet import CuePacket
from lmf.core.input.tokenizer import Vocabulary, build_vocabulary_from_texts
from lmf.core.state.types import ActiveRegion, BasinState
from lmf.training.basin_families import BasinFamilyExample, build_family_registry

LOGGER = logging.getLogger(__name__)


@dataclass
class BasinFieldStackConfig:
    num_traces: int = 64
    top_k: int = 16
    cue_dim: int = 16
    num_basins: int = 16
    relation_channels: int = 4
    settling_steps: int = 3
    contrastive_temperature: float = 0.1
    contrastive_weight: float = 0.05
    seed: int = 7

    def __post_init__(self) -> None:
        if self.num_traces <= 0:
            raise ValueError("num_traces must be positive.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.cue_dim <= 0:
            raise ValueError("cue_dim must be positive.")
        if self.num_basins <= 0:
            raise ValueError("num_basins must be positive.")
        if self.relation_channels <= 0:
            raise ValueError("relation_channels must be positive.")
        if self.settling_steps <= 0:
            raise ValueError("settling_steps must be positive.")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive.")
        if self.contrastive_weight < 0.0:
            raise ValueError("contrastive_weight must be non-negative.")


@dataclass
class BasinFieldForwardResult:
    text: str
    family: str
    family_index: int
    cue_packet: CuePacket
    active_region: ActiveRegion
    basin_state: BasinState
    basin_pressures: Tensor
    attractor_state: Tensor


class BasinFieldStack(nn.Module):
    """Text → cues → routing → field loop → basin attractor state → contrastive loss."""

    def __init__(
        self,
        config: BasinFieldStackConfig,
        vocab: Vocabulary,
        family_registry: dict[str, int],
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.family_registry = dict(family_registry)
        self.logger = logger

        self.cue_encoder = CueEncoder(
            CueEncoderConfig(
                vocab_size=len(vocab),
                cue_dim=config.cue_dim,
                pad_id=vocab.pad_id,
            )
        )
        self.trace_bank = TraceBank(
            TraceBankConfig(
                num_traces=config.num_traces,
                key_dim=config.cue_dim,
                content_dim=config.cue_dim,
            )
        )
        self.trace_router = TraceRouter(TraceRouterConfig(top_k=config.top_k))
        self.field_loop = FieldLoop(
            FieldLoopConfig(
                cue_dim=config.cue_dim,
                content_dim=config.cue_dim,
                active_traces=config.top_k,
                num_basins=config.num_basins,
                relation_channels=config.relation_channels,
                settling_steps=config.settling_steps,
            )
        )
        self.family_contrastive = BasinFamilyContrastive(
            BasinFamilyContrastiveConfig(
                num_basins=config.num_basins,
                basin_dim=config.cue_dim,
                temperature=config.contrastive_temperature,
                loss_weight=config.contrastive_weight,
            )
        )

    @classmethod
    def from_examples(
        cls,
        examples: list[BasinFamilyExample],
        config: BasinFieldStackConfig,
    ) -> BasinFieldStack:
        texts = [example.text for example in examples]
        vocab = build_vocabulary_from_texts(texts)
        registry = build_family_registry(examples)
        torch.manual_seed(config.seed)
        return cls(config, vocab, registry)

    def basin_vectors(self) -> Tensor:
        return self.field_loop.basin_bank.vectors

    def family_index_for(self, family: str) -> int:
        if family not in self.family_registry:
            raise ValueError(f"unknown basin family {family!r}; known: {sorted(self.family_registry)}")
        return self.family_registry[family]

    def set_bound_pair_to_basin_scale(self, scale: float) -> None:
        if scale < 0.0:
            raise ValueError("bound_pair_to_basin_scale must be non-negative.")
        self.field_loop.binding_forces.config.bound_pair_to_basin_scale = scale
        self.field_loop.binding_forces.basin_composer.config.bound_pair_to_basin_scale = scale

    def set_direct_trace_to_basin_scale(self, scale: float) -> None:
        if scale < 0.0:
            raise ValueError("direct_trace_to_basin_scale must be non-negative.")
        self.field_loop.binding_forces.config.direct_trace_to_basin_scale = scale
        self.field_loop.binding_forces.basin_composer.config.direct_trace_to_basin_scale = scale

    def set_settling_steps(self, steps: int) -> None:
        if steps <= 0:
            raise ValueError("settling_steps must be positive.")
        self.field_loop.config.settling_steps = steps

    def encode_example(self, example: BasinFamilyExample) -> tuple[CuePacket, ActiveRegion, BasinState]:
        from lmf.core.dmf.active_region import build_active_region
        from lmf.core.input.tokenizer import SimpleTokenizer

        tokenizer = SimpleTokenizer(vocab=self.vocab)
        ids = tokenizer.encode(example.text, add_bos=True, add_eos=True)
        input_ids = torch.tensor([ids], dtype=torch.long)
        cue_packet = self.cue_encoder(input_ids)
        cue_packet = CuePacket(
            cues=cue_packet.cues,
            mask=cue_packet.mask,
            positions=cue_packet.positions,
            pooled=cue_packet.pooled,
            token_ids=input_ids,
            special_token_ids=(
                self.vocab.pad_id,
                self.vocab.unk_id,
                self.vocab.mask_id,
                self.vocab.bos_id,
                self.vocab.eos_id,
            ),
        )
        routing = self.trace_router(cue_packet, self.trace_bank)
        active_region = build_active_region(
            trace_bank=self.trace_bank,
            routing=routing,
            cue_packet=cue_packet,
        )
        basin_state = self.field_loop.make_basin_state(
            batch_size=active_region.trace_amp.shape[0],
            device=active_region.trace_amp.device,
        )
        return cue_packet, active_region, basin_state

    def forward_example(self, example: BasinFamilyExample) -> BasinFieldForwardResult:
        cue_packet, active_region, basin_state = self.encode_example(example)
        field_output = self.field_loop(cue_packet, active_region, basin_state)
        vectors = self.basin_vectors()
        attractor = compute_basin_attractor_state(field_output.basin_pressures, vectors)
        return BasinFieldForwardResult(
            text=example.text,
            family=example.family,
            family_index=self.family_index_for(example.family),
            cue_packet=cue_packet,
            active_region=active_region,
            basin_state=basin_state,
            basin_pressures=field_output.basin_pressures,
            attractor_state=attractor,
        )

    def forward_batch(
        self,
        examples: list[BasinFamilyExample],
    ) -> tuple[Tensor, Tensor, Tensor, list[BasinFieldForwardResult]]:
        if len(examples) < 2:
            raise ValueError("contrastive training requires at least two examples per batch.")

        results = [self.forward_example(example) for example in examples]
        pressures = torch.cat([result.basin_pressures for result in results], dim=0)
        family_ids = [result.family_index for result in results]
        family_index = family_indices_tensor(
            family_ids,
            device=pressures.device,
        )
        vectors = self.basin_vectors()
        return pressures, vectors, family_index, results

    def contrastive_loss_from_batch(
        self,
        examples: list[BasinFamilyExample],
    ) -> tuple[Tensor, BasinFamilyContrastiveMetrics]:
        pressures, vectors, family_index, _results = self.forward_batch(examples)
        return self.family_contrastive(pressures, vectors, family_index)

    def separation_margin_for_examples(
        self,
        examples: list[BasinFamilyExample],
        *,
        basin_vectors: Tensor | None = None,
    ) -> float:
        """Evaluate family separation margin without building a loss graph."""

        self.eval()
        with torch.no_grad():
            pressures, vectors, family_index, _results = self.forward_batch(examples)
            readout_vectors = basin_vectors if basin_vectors is not None else vectors
            metrics = self.family_contrastive.metrics_only(
                pressures,
                readout_vectors,
                family_index,
            )
        return metrics.separation_margin


def build_basin_field_stack(
    examples: list[BasinFamilyExample],
    *,
    cue_dim: int = 16,
    num_traces: int = 64,
    top_k: int = 16,
    num_basins: int = 16,
    settling_steps: int = 3,
    contrastive_weight: float = 0.05,
    seed: int = 7,
) -> BasinFieldStack:
    return BasinFieldStack.from_examples(
        examples,
        BasinFieldStackConfig(
            cue_dim=cue_dim,
            num_traces=num_traces,
            top_k=top_k,
            num_basins=num_basins,
            settling_steps=settling_steps,
            contrastive_weight=contrastive_weight,
            seed=seed,
        ),
    )


__all__ = [
    "BasinFieldForwardResult",
    "BasinFieldStack",
    "BasinFieldStackConfig",
    "build_basin_field_stack",
]
