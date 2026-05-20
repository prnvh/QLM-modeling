"""End-to-end binding path: text → cues → routing → context-aware binding."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lmf.core.dmf.trace_bank import TraceBank, TraceBankConfig
from lmf.core.dmf.trace_router import TraceRouter, TraceRouterConfig
from lmf.core.field.binding import BindingLayer, BindingLayerConfig
from lmf.core.field.context_op import ContextOp, ContextOpConfig
from lmf.core.field.types import ContextPressure
from lmf.core.input.cue_encoder import CueEncoder, CueEncoderConfig
from lmf.core.input.cue_packet import CuePacket
from lmf.core.input.tokenizer import Vocabulary, build_vocabulary_from_texts
from lmf.core.state.types import ActiveRegion, BindingState
from lmf.training.binding_edges import BindingEdgeExample, ResolvedBindingEdge, resolve_example_edges

LOGGER = logging.getLogger(__name__)


@dataclass
class BindingStackConfig:
    num_traces: int = 64
    top_k: int = 8
    cue_dim: int = 16
    relation_channels: int = 4
    seed: int = 7

    def __post_init__(self) -> None:
        if self.num_traces <= 0:
            raise ValueError("num_traces must be positive.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.cue_dim <= 0:
            raise ValueError("cue_dim must be positive.")
        if self.relation_channels <= 0:
            raise ValueError("relation_channels must be positive.")


@dataclass
class BindingForwardResult:
    text: str
    cue_packet: CuePacket
    active_region: ActiveRegion
    context: ContextPressure
    pair_mass: Tensor
    binding_state: BindingState
    resolved_edges: tuple[ResolvedBindingEdge, ...]


class BindingStack(nn.Module):
    """Trainable stack for context-sensitive binding over routed traces."""

    def __init__(
        self,
        config: BindingStackConfig,
        vocab: Vocabulary,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab = vocab
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
        self.context_op = ContextOp(
            ContextOpConfig(
                cue_dim=config.cue_dim,
                active_traces=config.top_k,
                num_basins=1,
            )
        )
        self.binding_layer = BindingLayer(
            BindingLayerConfig(
                content_dim=config.cue_dim,
                context_dim=config.cue_dim,
                relation_channels=config.relation_channels,
            )
        )

    @classmethod
    def from_examples(
        cls,
        examples: list[BindingEdgeExample],
        config: BindingStackConfig,
    ) -> BindingStack:
        texts = [example.text for example in examples]
        vocab = build_vocabulary_from_texts(texts)
        torch.manual_seed(config.seed)
        return cls(config, vocab)

    def encode_example(self, example: BindingEdgeExample) -> tuple[CuePacket, ActiveRegion]:
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
        from lmf.core.dmf.active_region import build_active_region

        active_region = build_active_region(
            trace_bank=self.trace_bank,
            routing=routing,
            cue_packet=cue_packet,
        )
        return cue_packet, active_region

    def forward_example(self, example: BindingEdgeExample) -> BindingForwardResult:
        cue_packet, active_region = self.encode_example(example)
        context = self.context_op(cue_packet, active_region)
        pair_mass = self.binding_layer.pairwise_strength_matrix(active_region, context=context)
        binding_state = self.binding_layer(
            active_region,
            basin_state=_placeholder_basin(active_region),
            context=context,
        )
        resolved = resolve_example_edges(example)
        return BindingForwardResult(
            text=example.text,
            cue_packet=cue_packet,
            active_region=active_region,
            context=context,
            pair_mass=pair_mass,
            binding_state=binding_state,
            resolved_edges=resolved,
        )

    def binding_edge_loss(self, result: BindingForwardResult) -> Tensor:
        losses: list[Tensor] = []
        for edge in result.resolved_edges:
            indices_a = trace_indices_for_positions(result.active_region, edge.cue_a_positions)
            indices_b = trace_indices_for_positions(result.active_region, edge.cue_b_positions)
            if not indices_a or not indices_b:
                continue
            prediction = best_pair_mass(result.pair_mass, indices_a=indices_a, indices_b=indices_b)
            target = result.pair_mass.new_tensor(float(edge.label))
            losses.append(F.binary_cross_entropy(prediction, target))
        if not losses:
            return result.pair_mass.new_zeros(())
        return torch.stack(losses).mean()


def _placeholder_basin(active_region: ActiveRegion):
    from lmf.core.state.types import BasinState

    batch_size = active_region.trace_amp.shape[0]
    device = active_region.trace_amp.device
    dtype = active_region.trace_amp.dtype
    return BasinState(
        pressures=torch.zeros(batch_size, 1, device=device, dtype=dtype),
        vectors=torch.zeros(1, active_region.trace_content.shape[-1], device=device, dtype=dtype),
    )


def trace_indices_for_positions(active_region: ActiveRegion, positions: tuple[int, ...]) -> list[int]:
    if active_region.source_cue_id is None:
        return []
    source = active_region.source_cue_id[0].tolist()
    wanted = set(positions)
    return [index for index, cue_pos in enumerate(source) if int(cue_pos) in wanted]


def best_pair_mass(pair_mass: Tensor, *, indices_a: list[int], indices_b: list[int]) -> Tensor:
    best: Tensor | None = None
    for left in indices_a:
        for right in indices_b:
            if left == right:
                continue
            value = pair_mass[0, left, right]
            best = value if best is None else torch.maximum(best, value)
    if best is None:
        return pair_mass.new_zeros(())
    return best


def build_binding_stack(
    examples: list[BindingEdgeExample],
    *,
    cue_dim: int = 16,
    num_traces: int = 64,
    top_k: int = 16,
    seed: int = 7,
) -> BindingStack:
    return BindingStack.from_examples(
        examples,
        BindingStackConfig(
            num_traces=num_traces,
            top_k=top_k,
            cue_dim=cue_dim,
            seed=seed,
        ),
    )


__all__ = [
    "BindingForwardResult",
    "BindingStack",
    "BindingStackConfig",
    "best_pair_mass",
    "build_binding_stack",
    "trace_indices_for_positions",
]
