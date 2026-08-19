"""Sequential chunks for training persistent-state models."""

from __future__ import annotations

import random
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from neural_highlight.dataset.fragments import IGNORE_LABEL_ID, PAD_BYTE_ID
from neural_highlight.dataset.storage import IndexedJsonlStore, StoredFile
from neural_highlight.dataset.prose import sample_prose
from neural_highlight.languages import LANGUAGE_IDS
from neural_highlight.labels import Label


class StreamingFragmentDataset(Dataset[dict[str, Tensor]]):
    """Return ordered chunk sequences sampled from fully annotated files."""

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        chunk_length: int = 256,
        lookahead: int = 128,
        chunks_per_sample: int = 8,
        samples_per_epoch: int = 4096,
        seed: int = 0,
        start_at_file_beginning_fraction: float = 0.25,
        delimiter_wrap_fraction: float = 0.1,
        prose_code_fraction: float = 0.15,
    ) -> None:
        paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(path) for path in paths]
        self.store = IndexedJsonlStore(paths)
        if not len(self.store):
            raise ValueError("streaming dataset has no records")
        if chunk_length <= 0 or lookahead < 0 or chunks_per_sample <= 0:
            raise ValueError("invalid streaming dimensions")
        self.chunk_length = chunk_length
        self.lookahead = lookahead
        self.chunks_per_sample = chunks_per_sample
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.start_at_file_beginning_fraction = start_at_file_beginning_fraction
        self.delimiter_wrap_fraction = delimiter_wrap_fraction
        if not 0.0 <= prose_code_fraction <= 1.0:
            raise ValueError("prose_code_fraction must be between zero and one")
        self.prose_code_fraction = prose_code_fraction
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _region_labels(self, record: StoredFile) -> bytes:
        if record.region_labels is not None:
            return record.region_labels
        language = LANGUAGE_IDS.get(record.language.lower(), LANGUAGE_IDS["unknown"])
        return bytes([language]) * len(record.source)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = random.Random((self.seed << 64) ^ (self.epoch << 32) ^ index)
        record_index = rng.randrange(len(self.store))
        record = self.store[record_index]
        committed_total = self.chunk_length * self.chunks_per_sample
        language_id = LANGUAGE_IDS.get(record.language.lower(), LANGUAGE_IDS["unknown"])
        prose = (
            sample_prose(self.store, rng, max(16, committed_total // 2))
            if rng.random() < self.prose_code_fraction else None
        )
        if prose is not None:
            total_length = committed_total + self.lookahead
            opening, closing = rng.choice(((b"/* ", b" */"), (b"", b"")))
            prose = prose[: max(1, committed_total // 2)]
            prefix = opening + prose + b"\n"
            code_length = max(1, total_length - len(prefix) - len(closing))
            payload_start = rng.randrange(max(0, len(record.source) - code_length) + 1)
            code = record.source[payload_start : payload_start + code_length]
            original_regions = self._region_labels(record)
            original_mask = record.label_mask or bytes([1]) * len(record.source)
            source = prefix + code + closing
            labels_source = (
                bytes([Label.COMMENT]) * len(prefix)
                + record.labels[payload_start : payload_start + len(code)]
                + bytes([Label.COMMENT]) * len(closing)
            )
            regions = (
                bytes([language_id]) * len(prefix)
                + original_regions[payload_start : payload_start + len(code)]
                + bytes([language_id]) * len(closing)
            )
            supervision = (
                bytes([1]) * len(prefix)
                + original_mask[payload_start : payload_start + len(code)]
                + bytes([1]) * len(closing)
            )
            start = 0
        elif rng.random() < self.delimiter_wrap_fraction:
            total_length = committed_total + self.lookahead
            payload_length = max(1, total_length - 4)
            payload_start = rng.randrange(max(0, len(record.source) - payload_length) + 1)
            payload = record.source[payload_start : payload_start + payload_length]
            payload_labels = record.labels[payload_start : payload_start + len(payload)]
            original_regions = self._region_labels(record)
            payload_regions = original_regions[payload_start : payload_start + len(payload)]
            original_mask = record.label_mask or bytes([1]) * len(record.source)
            payload_mask = original_mask[payload_start : payload_start + len(payload)]
            source = b"/*" + payload + b"*/"
            labels_source = bytes([Label.COMMENT]) * 2 + payload_labels + bytes([Label.COMMENT]) * 2
            regions = bytes([language_id]) * 2 + payload_regions + bytes([language_id]) * 2
            supervision = b"\x01\x01" + payload_mask + b"\x01\x01"
            start = 0
        else:
            source = record.source
            labels_source = record.labels
            regions = self._region_labels(record)
            supervision = record.label_mask or bytes([1]) * len(record.source)
            largest_start = max(0, len(source) - committed_total)
            start = (
                0
                if rng.random() < self.start_at_file_beginning_fraction
                else rng.randrange(largest_start + 1)
            )
        inputs = []
        labels = []
        region_targets = []
        masks = []
        input_length = self.chunk_length + self.lookahead
        for chunk_index in range(self.chunks_per_sample):
            offset = start + chunk_index * self.chunk_length
            source_part = source[offset : offset + input_length]
            label_part = labels_source[offset : offset + self.chunk_length]
            region_part = regions[offset : offset + self.chunk_length]
            mask_part = supervision[offset : offset + self.chunk_length]
            input_array = np.full(input_length, PAD_BYTE_ID, dtype=np.int64)
            label_array = np.full(self.chunk_length, IGNORE_LABEL_ID, dtype=np.int64)
            region_array = np.full(self.chunk_length, IGNORE_LABEL_ID, dtype=np.int64)
            mask_array = np.zeros(self.chunk_length, dtype=np.bool_)
            input_array[: len(source_part)] = np.frombuffer(source_part, dtype=np.uint8)
            label_array[: len(label_part)] = np.frombuffer(label_part, dtype=np.uint8)
            region_array[: len(region_part)] = np.frombuffer(region_part, dtype=np.uint8)
            supervised = np.frombuffer(mask_part, dtype=np.uint8).astype(np.bool_)
            label_array[: len(label_part)][~supervised] = IGNORE_LABEL_ID
            region_array[: len(region_part)][~supervised] = IGNORE_LABEL_ID
            mask_array[: len(label_part)] = True
            inputs.append(torch.from_numpy(input_array))
            labels.append(torch.from_numpy(label_array))
            region_targets.append(torch.from_numpy(region_array))
            masks.append(torch.from_numpy(mask_array))
        return {
            "input_ids": torch.stack(inputs),
            "labels": torch.stack(labels),
            "region_labels": torch.stack(region_targets),
            "attention_mask": torch.stack(masks),
            "language_id": torch.tensor(
                language_id
            ),
            "record_index": torch.tensor(record_index),
            "start_byte": torch.tensor(start),
        }
