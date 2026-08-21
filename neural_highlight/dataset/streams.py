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
from neural_highlight.dataset.comment_wrapping import (
    random_comment_wrapper_style,
    wrap_code_as_comment,
)
from neural_highlight.dataset.string_wrapping import (
    random_string_wrapper_style,
    string_wrapper_languages,
    wrap_code_as_string,
)
from neural_highlight.languages import LANGUAGE_IDS
from neural_highlight.labels import Enclosure, Label


SAMPLE_KINDS = (
    "ordinary", "prose_code", "wrapped_code", "mixed_wrapped",
    "string_code", "long_string", "mixed_ordinary",
)
SAMPLE_KIND_IDS = {name: index for index, name in enumerate(SAMPLE_KINDS)}


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
        mixed_language_fraction: float = 0.0,
        string_code_fraction: float = 0.0,
        long_string_fraction: float = 0.0,
        mixed_ordinary_fraction: float = 0.0,
        embedded_loss_weight: float = 2.0,
        quota_cycle: int = 16,
    ) -> None:
        paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(path) for path in paths]
        self.store = IndexedJsonlStore(paths)
        if not len(self.store):
            raise ValueError("streaming dataset has no records")
        self.store.require_current_label_schema()
        if chunk_length <= 0 or lookahead < 0 or chunks_per_sample <= 0:
            raise ValueError("invalid streaming dimensions")
        self.chunk_length = chunk_length
        self.lookahead = lookahead
        self.chunks_per_sample = chunks_per_sample
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.start_at_file_beginning_fraction = start_at_file_beginning_fraction
        self.delimiter_wrap_fraction = delimiter_wrap_fraction
        if not 0.0 <= delimiter_wrap_fraction <= 1.0:
            raise ValueError("delimiter_wrap_fraction must be between zero and one")
        if not 0.0 <= prose_code_fraction <= 1.0:
            raise ValueError("prose_code_fraction must be between zero and one")
        self.prose_code_fraction = prose_code_fraction
        if not 0.0 <= mixed_language_fraction <= 1.0:
            raise ValueError("mixed_language_fraction must be between zero and one")
        self.mixed_language_fraction = mixed_language_fraction
        if not 0.0 <= string_code_fraction <= 1.0:
            raise ValueError("string_code_fraction must be between zero and one")
        self.string_code_fraction = string_code_fraction
        if not 0.0 <= long_string_fraction <= 1.0:
            raise ValueError("long_string_fraction must be between zero and one")
        self.long_string_fraction = long_string_fraction
        if not 0.0 <= mixed_ordinary_fraction <= 1.0:
            raise ValueError("mixed_ordinary_fraction must be between zero and one")
        self.mixed_ordinary_fraction = mixed_ordinary_fraction
        if embedded_loss_weight < 1.0:
            raise ValueError("embedded_loss_weight must be at least one")
        self.embedded_loss_weight = embedded_loss_weight
        if quota_cycle <= 0:
            raise ValueError("quota_cycle must be positive")
        self.quota_cycle = quota_cycle
        self.epoch = 0
        by_language: dict[str, list[int]] = {}
        for record_index in range(len(self.store)):
            by_language.setdefault(self.store.language(record_index), []).append(record_index)
        self._indices_by_language = {key: tuple(value) for key, value in by_language.items()}
        self._languages = tuple(by_language)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _region_labels(self, record: StoredFile) -> bytes:
        if record.region_labels is not None:
            return record.region_labels
        language = LANGUAGE_IDS.get(record.language.lower(), LANGUAGE_IDS["unknown"])
        return bytes([language]) * len(record.source)

    def _random_record_index(self, rng: random.Random) -> int:
        language = self._languages[0] if len(self._languages) == 1 else rng.choice(self._languages)
        indices = self._indices_by_language[language]
        return indices[0] if len(indices) == 1 else rng.choice(indices)

    def _sample_kind(self, index: int) -> str:
        """Assign stable per-batch quotas instead of probabilistic augmentation."""
        cycle = self.quota_cycle
        slot = index % cycle
        prose_slots = min(cycle, round(self.prose_code_fraction * cycle))
        wrapped_slots = min(cycle - prose_slots, round(self.delimiter_wrap_fraction * cycle))
        mixed_slots = min(
            cycle - prose_slots - wrapped_slots,
            round(self.mixed_language_fraction * cycle),
        )
        string_slots = min(
            cycle - prose_slots - wrapped_slots - mixed_slots,
            round(self.string_code_fraction * cycle),
        )
        long_string_slots = min(
            cycle - prose_slots - wrapped_slots - mixed_slots - string_slots,
            round(self.long_string_fraction * cycle),
        )
        mixed_ordinary_slots = min(
            cycle - prose_slots - wrapped_slots - mixed_slots - string_slots
            - long_string_slots,
            round(self.mixed_ordinary_fraction * cycle),
        )
        if slot < prose_slots:
            return "prose_code"
        if slot < prose_slots + wrapped_slots:
            return "wrapped_code"
        if slot < prose_slots + wrapped_slots + mixed_slots:
            return "mixed_wrapped"
        if slot < prose_slots + wrapped_slots + mixed_slots + string_slots:
            return "string_code"
        if slot < prose_slots + wrapped_slots + mixed_slots + string_slots + long_string_slots:
            return "long_string"
        if slot < (
            prose_slots + wrapped_slots + mixed_slots + string_slots + long_string_slots
            + mixed_ordinary_slots
        ):
            return "mixed_ordinary"
        return "ordinary"

    def _slice_record(
        self, record: StoredFile, rng: random.Random, length: int
    ) -> tuple[bytes, bytes, bytes, bytes]:
        start = rng.randrange(max(0, len(record.source) - length) + 1)
        source = record.source[start : start + length]
        mask = record.label_mask or bytes([1]) * len(record.source)
        return (
            source,
            record.labels[start : start + len(source)],
            self._region_labels(record)[start : start + len(source)],
            mask[start : start + len(source)],
        )

    def _mixed_payload(
        self, rng: random.Random, length: int
    ) -> tuple[bytes, bytes, bytes, bytes]:
        languages = rng.sample(self._languages, min(3, len(self._languages)))
        target_lengths = [length // len(languages)] * len(languages)
        target_lengths[-1] += length - sum(target_lengths)
        parts = [
            self._slice_record(
                self.store[rng.choice(self._indices_by_language[language])], rng, part_length
            )
            for language, part_length in zip(languages, target_lengths)
        ]
        sources: list[bytes] = []
        labels: list[bytes] = []
        regions: list[bytes] = []
        masks: list[bytes] = []
        for part_index, part in enumerate(parts):
            if part_index:
                sources.append(b"\n")
                labels.append(bytes([Label.PLAIN]))
                regions.append(bytes([LANGUAGE_IDS["unknown"]]))
                masks.append(b"\x01")
            sources.append(part[0])
            labels.append(part[1])
            regions.append(part[2])
            masks.append(part[3])
        return b"".join(sources), b"".join(labels), b"".join(regions), b"".join(masks)

    def _prose_code_payload(
        self, record: StoredFile, prose: bytes, rng: random.Random, length: int
    ) -> tuple[bytes, bytes, bytes, bytes]:
        half = max(1, len(prose) // 2)
        prose_parts = (prose[:half], prose[half:])
        code_length = max(1, (length - len(prose) - 3) // 2)
        first = self._slice_record(record, rng, code_length)
        second = self._slice_record(record, rng, code_length)
        prose_region = bytes([LANGUAGE_IDS["prose"]])
        comment = bytes([Label.COMMENT])
        sources = (prose_parts[0], b"\n", first[0], b"\n", prose_parts[1], b"\n", second[0])
        labels = (
            comment * len(prose_parts[0]), comment, first[1], comment,
            comment * len(prose_parts[1]), comment, second[1],
        )
        regions = (
            prose_region * len(prose_parts[0]), prose_region, first[2], prose_region,
            prose_region * len(prose_parts[1]), prose_region, second[2],
        )
        masks = (
            bytes([1]) * len(prose_parts[0]), b"\x01", first[3], b"\x01",
            bytes([1]) * len(prose_parts[1]), b"\x01", second[3],
        )
        return (
            b"".join(sources), b"".join(labels), b"".join(regions), b"".join(masks)
        )

    def _string_code_sample(
        self, rng: random.Random, length: int
    ) -> tuple[bytes, bytes, bytes, bytes, int]:
        available_hosts = [
            language for language in string_wrapper_languages()
            if language in self._indices_by_language
        ]
        host_language = rng.choice(available_hosts)
        payload_languages = [language for language in self._languages if language != host_language]
        payload_language = rng.choice(payload_languages)
        record = self.store[rng.choice(self._indices_by_language[payload_language])]
        prose = sample_prose(self.store, rng, max(16, length // 3))
        if prose is None:
            payload = self._slice_record(record, rng, length)
        else:
            half = max(1, len(prose) // 2)
            prose_parts = (prose[:half], prose[half:])
            code_length = max(1, (length - len(prose) - 3) // 2)
            first = self._slice_record(record, rng, code_length)
            second = self._slice_record(record, rng, code_length)
            string = bytes([Label.STRING])
            prose_region = bytes([LANGUAGE_IDS["prose"]])
            payload = (
                prose_parts[0] + b"\n" + first[0] + b"\n" + prose_parts[1] + b"\n" + second[0],
                string * len(prose_parts[0]) + string + first[1] + string
                + string * len(prose_parts[1]) + string + second[1],
                prose_region * len(prose_parts[0]) + prose_region + first[2] + prose_region
                + prose_region * len(prose_parts[1]) + prose_region + second[2],
                bytes([1]) * len(prose_parts[0]) + b"\x01" + first[3] + b"\x01"
                + bytes([1]) * len(prose_parts[1]) + b"\x01" + second[3],
            )
        host_region = LANGUAGE_IDS[host_language]
        wrapped = wrap_code_as_string(
            *payload, length,
            style=random_string_wrapper_style(host_language, rng),
            host_region=host_region,
        )
        return wrapped.source, wrapped.labels, wrapped.regions, wrapped.mask, host_region

    def _long_string_sample(
        self, rng: random.Random, length: int
    ) -> tuple[bytes, bytes, bytes, bytes, int]:
        available_hosts = [
            language for language in string_wrapper_languages()
            if language in self._indices_by_language
        ]
        host_language = rng.choice(available_hosts)
        prose = sample_prose(self.store, rng, max(16, length // 2))
        if not prose:
            prose = b"The configured value is described here in ordinary language.\n"
        payload = (prose.rstrip() + b"\n") * (length // max(1, len(prose)) + 2)
        host_region = LANGUAGE_IDS[host_language]
        wrapped = wrap_code_as_string(
            payload,
            bytes([Label.STRING]) * len(payload),
            bytes([host_region]) * len(payload),
            bytes([1]) * len(payload),
            length,
            style=random_string_wrapper_style(host_language, rng),
            host_region=host_region,
        )
        return wrapped.source, wrapped.labels, wrapped.regions, wrapped.mask, host_region

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = random.Random((self.seed << 64) ^ (self.epoch << 32) ^ index)
        record_index = self._random_record_index(rng)
        record = self.store[record_index]
        committed_total = self.chunk_length * self.chunks_per_sample
        language_id = LANGUAGE_IDS.get(record.language.lower(), LANGUAGE_IDS["unknown"])
        sample_kind = self._sample_kind(index)
        prose = (
            sample_prose(self.store, rng, max(16, committed_total // 2))
            if sample_kind == "prose_code" else None
        )
        if sample_kind == "prose_code" and prose is None:
            sample_kind = "wrapped_code"
        if prose is not None:
            total_length = committed_total + self.lookahead
            prose = prose[: max(1, committed_total // 2)]
            payload = self._prose_code_payload(record, prose, rng, total_length)
            wrapped = wrap_code_as_comment(
                *payload, total_length, style=random_comment_wrapper_style(record.language, rng)
            )
            source, labels_source, regions, supervision = (
                wrapped.source, wrapped.labels, wrapped.regions, wrapped.mask
            )
            loss_weights_source = bytes(
                round(self.embedded_loss_weight) if label != Label.COMMENT else 1
                for label in wrapped.labels
            )
            language_id = LANGUAGE_IDS["unknown"]
            start = 0
        elif sample_kind == "string_code":
            total_length = committed_total + self.lookahead
            source, labels_source, regions, supervision, language_id = self._string_code_sample(
                rng, total_length
            )
            loss_weights_source = bytes(
                round(self.embedded_loss_weight) if label != Label.STRING else 1
                for label in labels_source
            )
            start = 0
        elif sample_kind == "long_string":
            total_length = committed_total + self.lookahead
            source, labels_source, regions, supervision, language_id = (
                self._long_string_sample(rng, total_length)
            )
            loss_weights_source = bytes([1]) * len(source)
            start = 0
        elif sample_kind == "mixed_ordinary":
            total_length = committed_total + self.lookahead
            payload = self._mixed_payload(rng, total_length)
            source, labels_source, regions, supervision = (
                value[:total_length] for value in payload
            )
            loss_weights_source = bytes([1]) * len(source)
            language_id = LANGUAGE_IDS["unknown"]
            start = 0
        elif sample_kind in ("wrapped_code", "mixed_wrapped"):
            total_length = committed_total + self.lookahead
            payload = (
                self._mixed_payload(rng, total_length)
                if sample_kind == "mixed_wrapped"
                else self._slice_record(record, rng, total_length)
            )
            style = (
                rng.choice(("c_multiline", "c_starred"))
                if sample_kind == "mixed_wrapped"
                else random_comment_wrapper_style(record.language, rng)
            )
            wrapped = wrap_code_as_comment(
                *payload, total_length, style=style,
            )
            source = wrapped.source
            labels_source = wrapped.labels
            regions = wrapped.regions
            supervision = wrapped.mask
            loss_weights_source = bytes(
                round(self.embedded_loss_weight) if label != Label.COMMENT else 1
                for label in wrapped.labels
            )
            if sample_kind == "mixed_wrapped":
                language_id = LANGUAGE_IDS["unknown"]
            start = 0
        else:
            source = record.source
            labels_source = record.labels
            regions = self._region_labels(record)
            supervision = record.label_mask or bytes([1]) * len(record.source)
            loss_weights_source = bytes([1]) * len(source)
            largest_start = max(0, len(source) - committed_total)
            start = (
                0
                if rng.random() < self.start_at_file_beginning_fraction
                else rng.randrange(largest_start + 1)
            )
        enclosed = (
            Enclosure.STRING if sample_kind in ("string_code", "long_string")
            else Enclosure.COMMENT if sample_kind in (
                "prose_code", "wrapped_code", "mixed_wrapped"
            ) else None
        )
        enclosure_source = bytes(
            enclosed if enclosed is not None
            else Enclosure.STRING if label == Label.STRING
            else Enclosure.COMMENT if label == Label.COMMENT
            else Enclosure.CODE
            for label in labels_source
        )
        labels_source = bytes(
            Label.PLAIN if label in (Label.STRING, Label.COMMENT) else label
            for label in labels_source
        )
        inputs = []
        labels = []
        region_targets = []
        masks = []
        loss_weights = []
        enclosure_targets = []
        input_length = self.chunk_length + self.lookahead
        for chunk_index in range(self.chunks_per_sample):
            offset = start + chunk_index * self.chunk_length
            source_part = source[offset : offset + input_length]
            label_part = labels_source[offset : offset + self.chunk_length]
            region_part = regions[offset : offset + self.chunk_length]
            mask_part = supervision[offset : offset + self.chunk_length]
            weight_part = loss_weights_source[offset : offset + self.chunk_length]
            enclosure_part = enclosure_source[offset : offset + self.chunk_length]
            input_array = np.full(input_length, PAD_BYTE_ID, dtype=np.int64)
            label_array = np.full(self.chunk_length, IGNORE_LABEL_ID, dtype=np.int64)
            region_array = np.full(self.chunk_length, IGNORE_LABEL_ID, dtype=np.int64)
            mask_array = np.zeros(self.chunk_length, dtype=np.bool_)
            weight_array = np.ones(self.chunk_length, dtype=np.float32)
            enclosure_array = np.full(self.chunk_length, IGNORE_LABEL_ID, dtype=np.int64)
            input_array[: len(source_part)] = np.frombuffer(source_part, dtype=np.uint8)
            label_array[: len(label_part)] = np.frombuffer(label_part, dtype=np.uint8)
            region_array[: len(region_part)] = np.frombuffer(region_part, dtype=np.uint8)
            supervised = np.frombuffer(mask_part, dtype=np.uint8).astype(np.bool_)
            label_array[: len(label_part)][~supervised] = IGNORE_LABEL_ID
            region_array[: len(region_part)][~supervised] = IGNORE_LABEL_ID
            mask_array[: len(label_part)] = True
            weight_array[: len(weight_part)] = np.frombuffer(weight_part, dtype=np.uint8)
            enclosure_array[: len(enclosure_part)] = np.frombuffer(
                enclosure_part, dtype=np.uint8
            )
            inputs.append(torch.from_numpy(input_array))
            labels.append(torch.from_numpy(label_array))
            region_targets.append(torch.from_numpy(region_array))
            masks.append(torch.from_numpy(mask_array))
            loss_weights.append(torch.from_numpy(weight_array))
            enclosure_targets.append(torch.from_numpy(enclosure_array))
        return {
            "input_ids": torch.stack(inputs),
            "labels": torch.stack(labels),
            "region_labels": torch.stack(region_targets),
            "attention_mask": torch.stack(masks),
            "loss_weights": torch.stack(loss_weights),
            "enclosure_labels": torch.stack(enclosure_targets),
            "language_id": torch.tensor(
                language_id
            ),
            "record_index": torch.tensor(record_index),
            "start_byte": torch.tensor(start),
            "sample_kind_id": torch.tensor(SAMPLE_KIND_IDS[sample_kind]),
        }
