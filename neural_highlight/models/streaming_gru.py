"""Streaming GRU with persistent left state and bounded right lookahead."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from neural_highlight.dataset.fragments import PAD_BYTE_ID
from neural_highlight.labels import ENCLOSURE_NAMES, LABEL_NAMES
from neural_highlight.languages import LANGUAGE_NAMES


@dataclass(frozen=True)
class StreamingGRUConfig:
    byte_embedding_dim: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    use_language_embedding: bool = True
    language_embedding_dim: int = 8
    num_languages: int = len(LANGUAGE_NAMES)
    num_classes: int = len(LABEL_NAMES)
    num_enclosures: int = len(ENCLOSURE_NAMES)
    region_conditioned_syntax: bool = False
    detach_region_for_syntax: bool = True
    token_context_dim: int = 0
    token_context_radius: int = 2

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class StreamingByteGRU(nn.Module):
    """Classify committed bytes while carrying state across arbitrary chunks.

    ``input_ids`` contains committed bytes followed by optional lookahead.
    Persistent forward state advances through committed bytes only. A separate
    reverse GRU reads the complete input and supplies bounded future context.
    """

    def __init__(self, config: StreamingGRUConfig | None = None) -> None:
        super().__init__()
        self.config = config or StreamingGRUConfig()
        self.byte_embedding = nn.Embedding(
            PAD_BYTE_ID + 1, self.config.byte_embedding_dim, padding_idx=PAD_BYTE_ID
        )
        input_size = self.config.byte_embedding_dim
        if self.config.use_language_embedding:
            self.language_embedding = nn.Embedding(
                self.config.num_languages, self.config.language_embedding_dim
            )
            input_size += self.config.language_embedding_dim
        else:
            self.language_embedding = None
        dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.forward_encoder = nn.GRU(
            input_size, self.config.hidden_size, self.config.num_layers,
            batch_first=True, dropout=dropout,
        )
        self.reverse_encoder = nn.GRU(
            input_size, self.config.hidden_size, self.config.num_layers,
            batch_first=True, dropout=dropout,
        )
        representation_size = self.config.hidden_size * 2
        syntax_size = representation_size + self.config.token_context_dim
        self.syntax_classifier = nn.Linear(syntax_size, self.config.num_classes)
        self.token_context_projection = (
            nn.Linear(representation_size, self.config.token_context_dim)
            if self.config.token_context_dim > 0 else None
        )
        self.region_classifier = nn.Linear(representation_size, self.config.num_languages)
        self.enclosure_classifier = nn.Linear(
            representation_size, self.config.num_enclosures
        )
        if self.config.region_conditioned_syntax:
            self.region_syntax_weight = nn.Parameter(torch.zeros(
                self.config.num_languages, syntax_size, self.config.num_classes
            ))
            self.region_syntax_bias = nn.Parameter(torch.zeros(
                self.config.num_languages, self.config.num_classes
            ))
        else:
            self.register_parameter("region_syntax_weight", None)
            self.register_parameter("region_syntax_bias", None)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(
            self.config.num_layers, batch_size, self.config.hidden_size,
            device=device or self.byte_embedding.weight.device,
        )

    def _embed(self, input_ids: Tensor, language_id: Tensor | None) -> Tensor:
        embedded = self.byte_embedding(input_ids)
        if self.language_embedding is not None:
            if language_id is None:
                language_id = torch.zeros(
                    input_ids.shape[0], dtype=torch.long, device=input_ids.device
                )
            language = self.language_embedding(language_id).unsqueeze(1).expand(
                -1, input_ids.shape[1], -1
            )
            embedded = torch.cat((embedded, language), dim=-1)
        return embedded

    def _syntax_representation(self, representation: Tensor, input_ids: Tensor) -> Tensor:
        """Broadcast neighboring lexical-run context back to committed bytes."""
        if self.token_context_projection is None:
            return representation
        byte_ids = input_ids[:, : representation.shape[1]]
        word = (
            ((byte_ids >= ord("a")) & (byte_ids <= ord("z")))
            | ((byte_ids >= ord("A")) & (byte_ids <= ord("Z")))
            | (byte_ids == ord("_"))
            | ((byte_ids >= 128) & (byte_ids <= 255))
        )
        previous = torch.cat((torch.zeros_like(word[:, :1]), word[:, :-1]), dim=1)
        run_ids = ((word & ~previous).long().cumsum(dim=1) * word.long())
        batch, length, width = representation.shape
        sums = representation.new_zeros((batch, length + 1, width))
        sums.scatter_add_(1, run_ids.unsqueeze(-1).expand(-1, -1, width), representation)
        counts = representation.new_zeros((batch, length + 1, 1))
        counts.scatter_add_(1, run_ids.unsqueeze(-1), word.unsqueeze(-1).to(representation.dtype))
        pooled = sums / counts.clamp_min(1)
        neighboring = torch.zeros_like(pooled)
        neighboring_count = torch.zeros_like(counts)
        indices = torch.arange(length + 1, device=representation.device).view(1, -1, 1)
        maximum = run_ids.max(dim=1, keepdim=True).values.unsqueeze(-1)
        for offset in range(-self.config.token_context_radius, self.config.token_context_radius + 1):
            candidate = indices + offset
            valid = (candidate > 0) & (candidate <= maximum)
            gather = candidate.clamp(0, length).expand(batch, -1, width)
            neighboring += pooled.gather(1, gather) * valid
            neighboring_count += valid
        neighboring = neighboring / neighboring_count.clamp_min(1)
        context = neighboring.gather(
            1, run_ids.unsqueeze(-1).expand(-1, -1, width)
        ) * word.unsqueeze(-1)
        return torch.cat((representation, self.token_context_projection(context)), dim=-1)

    def _syntax_logits(
        self, representation: Tensor, region_logits: Tensor, input_ids: Tensor,
    ) -> Tensor:
        syntax_representation = self._syntax_representation(representation, input_ids)
        syntax_logits = self.syntax_classifier(syntax_representation)
        if self.region_syntax_weight is not None and self.region_syntax_bias is not None:
            region_probabilities = region_logits.softmax(dim=-1)
            if self.config.detach_region_for_syntax:
                region_probabilities = region_probabilities.detach()
            adapter_logits = torch.einsum(
                "bth,lhc->btlc", syntax_representation, self.region_syntax_weight
            ) + self.region_syntax_bias[None, None]
            syntax_logits = syntax_logits + (
                region_probabilities.unsqueeze(-1) * adapter_logits
            ).sum(dim=2)
        return syntax_logits

    def forward_chunk(
        self,
        input_ids: Tensor,
        state_in: Tensor,
        commit_length: int,
        language_id: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if commit_length <= 0 or commit_length > input_ids.shape[1]:
            raise ValueError("commit_length must be within the input sequence")
        embedded = self._embed(input_ids, language_id)
        forward, state_out = self.forward_encoder(embedded[:, :commit_length], state_in)
        reversed_input = torch.flip(embedded, dims=(1,))
        reverse, _ = self.reverse_encoder(reversed_input)
        reverse = torch.flip(reverse, dims=(1,))[:, :commit_length]
        representation = torch.cat((forward, reverse), dim=-1)
        region_logits = self.region_classifier(representation)
        syntax_logits = self._syntax_logits(representation, region_logits, input_ids)
        return syntax_logits, region_logits, state_out

    def forward_chunk_with_aux(
        self, input_ids: Tensor, state_in: Tensor, commit_length: int,
        language_id: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return public predictions plus the training-only enclosure logits."""
        embedded = self._embed(input_ids, language_id)
        forward, state_out = self.forward_encoder(embedded[:, :commit_length], state_in)
        reverse, _ = self.reverse_encoder(torch.flip(embedded, dims=(1,)))
        representation = torch.cat(
            (forward, torch.flip(reverse, dims=(1,))[:, :commit_length]), dim=-1
        )
        region_logits = self.region_classifier(representation)
        syntax_logits = self._syntax_logits(representation, region_logits, input_ids)
        return syntax_logits, region_logits, self.enclosure_classifier(representation), state_out

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
