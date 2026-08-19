"""Streaming GRU with persistent left state and bounded right lookahead."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from neural_highlight.dataset.fragments import PAD_BYTE_ID
from neural_highlight.labels import LABEL_NAMES
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
        self.syntax_classifier = nn.Linear(representation_size, self.config.num_classes)
        self.region_classifier = nn.Linear(representation_size, self.config.num_languages)

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
        return (
            self.syntax_classifier(representation),
            self.region_classifier(representation),
            state_out,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

