"""Byte-level bidirectional GRU syntax classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from neural_highlight.dataset.fragments import PAD_BYTE_ID
from neural_highlight.labels import LABEL_NAMES
from neural_highlight.languages import LANGUAGE_NAMES


@dataclass(frozen=True)
class BiGRUConfig:
    byte_embedding_dim: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    use_language_embedding: bool = False
    language_embedding_dim: int = 8
    num_languages: int = len(LANGUAGE_NAMES)
    num_classes: int = len(LABEL_NAMES)

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class ByteBiGRU(nn.Module):
    def __init__(self, config: BiGRUConfig | None = None) -> None:
        super().__init__()
        self.config = config or BiGRUConfig()
        self.byte_embedding = nn.Embedding(PAD_BYTE_ID + 1, self.config.byte_embedding_dim, padding_idx=PAD_BYTE_ID)
        input_size = self.config.byte_embedding_dim
        if self.config.use_language_embedding:
            self.language_embedding = nn.Embedding(
                self.config.num_languages, self.config.language_embedding_dim
            )
            input_size += self.config.language_embedding_dim
        else:
            self.language_embedding = None
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.num_classes)
        self.region_classifier = nn.Linear(self.config.hidden_size * 2, self.config.num_languages)

    def forward(self, input_ids: Tensor, language_id: Tensor | None = None) -> Tensor:
        embedded = self.byte_embedding(input_ids)
        if self.language_embedding is not None:
            if language_id is None:
                language_id = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
            language = self.language_embedding(language_id).unsqueeze(1).expand(-1, input_ids.shape[1], -1)
            embedded = torch.cat((embedded, language), dim=-1)
        encoded, _ = self.encoder(embedded)
        return self.classifier(encoded)

    def forward_with_regions(
        self, input_ids: Tensor, language_id: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        embedded = self.byte_embedding(input_ids)
        if self.language_embedding is not None:
            if language_id is None:
                language_id = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
            language = self.language_embedding(language_id).unsqueeze(1).expand(-1, input_ids.shape[1], -1)
            embedded = torch.cat((embedded, language), dim=-1)
        encoded, _ = self.encoder(embedded)
        return self.classifier(encoded), self.region_classifier(encoded)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
