"""PyTorch dataset for fixed-length fragments of fully annotated files."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from neural_highlight.dataset.annotate import Annotation, colored, iter_spans
from neural_highlight.dataset.storage import IndexedJsonlStore, StoredFile
from neural_highlight.labels import Label, label_name
from neural_highlight.languages import LANGUAGE_IDS, LANGUAGE_NAMES


PAD_BYTE_ID = 256
IGNORE_LABEL_ID = -100


@dataclass(frozen=True)
class Fragment:
    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    language_id: Tensor
    region_labels: Tensor
    source_length: Tensor
    file_index: Tensor
    start_byte: Tensor


class FragmentDataset(Dataset[dict[str, Tensor]]):
    """Sample deterministic random crops from complete annotated files.

    ``targeted_fraction`` controls how often a crop is centered near a randomly
    selected non-plain byte. Calling ``set_epoch`` changes crops while preserving
    reproducibility across runs and DataLoader worker counts.
    """

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        fragment_length: int = 256,
        samples_per_epoch: int | None = None,
        targeted_fraction: float = 0.3,
        seed: int = 0,
        mixture_fraction: float = 0.25,
        max_regions: int = 4,
        delimiter_wrap_fraction: float = 0.1,
    ) -> None:
        if fragment_length <= 0:
            raise ValueError("fragment_length must be positive")
        if not 0.0 <= targeted_fraction <= 1.0:
            raise ValueError("targeted_fraction must be between zero and one")
        if not 0.0 <= mixture_fraction <= 1.0:
            raise ValueError("mixture_fraction must be between zero and one")
        if max_regions < 2:
            raise ValueError("max_regions must be at least two")
        if not 0.0 <= delimiter_wrap_fraction <= 1.0:
            raise ValueError("delimiter_wrap_fraction must be between zero and one")
        paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(item) for item in path]
        self.paths = tuple(paths)
        self.store = IndexedJsonlStore(paths)
        if not len(self.store):
            raise ValueError(f"no records found in {paths}")
        self.fragment_length = fragment_length
        self.samples_per_epoch = samples_per_epoch or len(self.store)
        self.targeted_fraction = targeted_fraction
        self.seed = seed
        self.mixture_fraction = mixture_fraction
        self.max_regions = max_regions
        self.delimiter_wrap_fraction = delimiter_wrap_fraction
        # Shared memory keeps persistent DataLoader workers synchronized when
        # the trainer advances to a new epoch.
        self._epoch = torch.zeros((), dtype=torch.long)
        try:
            self._epoch.share_memory_()
            self.epoch_is_shared = True
        except RuntimeError:
            # Restricted environments may block torch_shm_manager. The trainer
            # detects this and restarts workers each epoch instead.
            self.epoch_is_shared = False
        by_language: dict[str, list[int]] = {}
        for record_index in range(len(self.store)):
            by_language.setdefault(self.store.language(record_index), []).append(record_index)
        self._indices_by_language = {key: tuple(value) for key, value in by_language.items()}
        self._languages = tuple(by_language)

    @property
    def record_count(self) -> int:
        return len(self.store)

    def record_at(self, index: int) -> StoredFile:
        return self.store[index]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(epoch)

    def _rng(self, index: int) -> random.Random:
        return random.Random((self.seed << 64) ^ (int(self._epoch) << 32) ^ index)

    def _crop_start(self, record: StoredFile, rng: random.Random) -> int:
        largest_start = max(0, len(record.source) - self.fragment_length)
        if record.labels and rng.random() < self.targeted_fraction:
            # Rejection sampling avoids retaining every non-plain byte offset.
            # Syntax-labeled bytes are common, so this normally succeeds quickly.
            for _ in range(32):
                center = rng.randrange(len(record.labels))
                supervised = record.label_mask is None or record.label_mask[center]
                if supervised and record.labels[center] != Label.PLAIN:
                    jitter = rng.randrange(
                        -self.fragment_length // 4, self.fragment_length // 4 + 1
                    )
                    return min(
                        largest_start,
                        max(0, center - self.fragment_length // 2 + jitter),
                    )
        return rng.randrange(largest_start + 1)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = self._rng(index)
        if self.fragment_length >= 5 and rng.random() < self.delimiter_wrap_fraction:
            return self._wrapped_code_sample(rng)
        if rng.random() < self.mixture_fraction:
            return self._mixed_sample(rng)
        return self._single_sample(rng)

    def _region_bytes(self, record: StoredFile, start: int, length: int) -> bytes:
        if record.region_labels is not None:
            return record.region_labels[start : start + length]
        language_id = LANGUAGE_IDS.get(record.language.lower(), LANGUAGE_IDS["unknown"])
        return bytes([language_id]) * length

    def _pack(
        self,
        source: bytes,
        labels: bytes,
        region_labels: bytes,
        label_mask: bytes,
        host_language_id: int,
        record_index: int,
        start: int,
    ) -> dict[str, Tensor]:
        source_length = len(source)
        input_values = np.full(self.fragment_length, PAD_BYTE_ID, dtype=np.int64)
        label_values = np.full(self.fragment_length, IGNORE_LABEL_ID, dtype=np.int64)
        region_values = np.full(self.fragment_length, IGNORE_LABEL_ID, dtype=np.int64)
        attention_values = np.zeros(self.fragment_length, dtype=np.bool_)
        input_values[:source_length] = np.frombuffer(source, dtype=np.uint8)
        label_values[:source_length] = np.frombuffer(labels, dtype=np.uint8)
        region_values[:source_length] = np.frombuffer(region_labels, dtype=np.uint8)
        supervision = np.frombuffer(label_mask, dtype=np.uint8).astype(np.bool_)
        label_values[:source_length][~supervision] = IGNORE_LABEL_ID
        region_values[:source_length][~supervision] = IGNORE_LABEL_ID
        attention_values[:source_length] = True
        return {
            "input_ids": torch.from_numpy(input_values),
            "labels": torch.from_numpy(label_values),
            "region_labels": torch.from_numpy(region_values),
            "attention_mask": torch.from_numpy(attention_values),
            "language_id": torch.tensor(host_language_id, dtype=torch.long),
            "source_length": torch.tensor(source_length, dtype=torch.long),
            "file_index": torch.tensor(record_index, dtype=torch.long),
            "start_byte": torch.tensor(start, dtype=torch.long),
        }

    def _single_sample(self, rng: random.Random) -> dict[str, Tensor]:
        record_index = rng.randrange(len(self.store))
        record = self.store[record_index]
        start = self._crop_start(record, rng)
        source = record.source[start : start + self.fragment_length]
        labels = record.labels[start : start + self.fragment_length]
        language_id = LANGUAGE_IDS.get(record.language.lower(), LANGUAGE_IDS["unknown"])
        return self._pack(
            source, labels, self._region_bytes(record, start, len(source)),
            (
                record.label_mask[start : start + len(source)]
                if record.label_mask is not None
                else bytes([1]) * len(source)
            ),
            language_id, record_index, start,
        )

    def _mixed_sample(self, rng: random.Random) -> dict[str, Tensor]:
        region_count = rng.randint(2, min(self.max_regions, self.fragment_length))
        boundaries = sorted(rng.sample(range(1, self.fragment_length), region_count - 1))
        lengths = [end - start for start, end in zip((0, *boundaries), (*boundaries, self.fragment_length))]
        sources: list[bytes] = []
        labels: list[bytes] = []
        regions: list[bytes] = []
        masks: list[bytes] = []
        first_record_index = 0
        previous_language: str | None = None
        for region_index, length in enumerate(lengths):
            language_choices = [
                language for language in self._languages if language != previous_language
            ] or list(self._languages)
            language = rng.choice(language_choices)
            record_index = rng.choice(self._indices_by_language[language])
            if region_index == 0:
                first_record_index = record_index
            record = self.store[record_index]
            previous_language = record.language
            largest_start = max(0, len(record.source) - length)
            start = rng.randrange(largest_start + 1)
            part_source = record.source[start : start + length]
            sources.append(part_source)
            labels.append(record.labels[start : start + length])
            regions.append(self._region_bytes(record, start, len(part_source)))
            masks.append(
                record.label_mask[start : start + len(part_source)]
                if record.label_mask is not None
                else bytes([1]) * len(part_source)
            )
        return self._pack(
            b"".join(sources), b"".join(labels), b"".join(regions), b"".join(masks),
            LANGUAGE_IDS["unknown"], first_record_index, -1,
        )

    def _wrapped_code_sample(self, rng: random.Random) -> dict[str, Tensor]:
        """Wrap genuine code in comment markers without erasing its syntax labels."""
        record_index = rng.randrange(len(self.store))
        record = self.store[record_index]
        delimiter_pairs = [
            pair for pair in ((b"/*", b"*/"), (b"<!--", b"-->"))
            if len(pair[0]) + len(pair[1]) < self.fragment_length
        ]
        opening, closing = rng.choice(delimiter_pairs)
        payload_length = self.fragment_length - len(opening) - len(closing)
        largest_start = max(0, len(record.source) - payload_length)
        start = rng.randrange(largest_start + 1)
        payload = record.source[start : start + payload_length]
        payload_labels = record.labels[start : start + len(payload)]
        payload_regions = self._region_bytes(record, start, len(payload))
        payload_mask = (
            record.label_mask[start : start + len(payload)]
            if record.label_mask is not None
            else bytes([1]) * len(payload)
        )
        language_id = LANGUAGE_IDS.get(record.language.lower(), LANGUAGE_IDS["unknown"])
        delimiter_labels = bytes([Label.COMMENT])
        return self._pack(
            opening + payload + closing,
            delimiter_labels * len(opening) + payload_labels + delimiter_labels * len(closing),
            bytes([language_id]) * len(opening) + payload_regions + bytes([language_id]) * len(closing),
            bytes([1]) * len(opening) + payload_mask + bytes([1]) * len(closing),
            language_id,
            record_index,
            -1,
        )

    def source_for(self, sample: dict[str, Tensor]) -> bytes:
        length = int(sample["source_length"])
        return bytes(sample["input_ids"][:length].tolist())


def describe_sample(dataset: FragmentDataset, index: int) -> str:
    sample = dataset[index]
    length = int(sample["source_length"])
    source = dataset.source_for(sample)
    raw_labels = sample["labels"][:length].tolist()
    labels = bytes(Label.PLAIN if value == IGNORE_LABEL_ID else value for value in raw_labels)
    region_labels = [
        LANGUAGE_IDS["unknown"] if value == IGNORE_LABEL_ID else value
        for value in sample["region_labels"][:length].tolist()
    ]
    annotation = Annotation(source, labels, ())
    record = dataset.record_at(int(sample["file_index"]))
    transitions = []
    if region_labels:
        start = 0
        for end in range(1, len(region_labels) + 1):
            if end == len(region_labels) or region_labels[end] != region_labels[start]:
                transitions.append(f"{start}:{end}={LANGUAGE_NAMES[region_labels[start]]}")
                start = end
    header = (
        f"{record.repository}/{record.path} bytes {int(sample['start_byte'])}:"
        f"{int(sample['start_byte']) + length} ({length}/{dataset.fragment_length})\n"
        f"regions: {', '.join(transitions)}"
    )
    legend = "\n".join(
        f"{start:4}:{end:<4} {label_name(label):20} {text!r}"
        for start, end, label, text in iter_spans(annotation)
        if label is not Label.PLAIN
    )
    return f"{header}\n{colored(annotation)}\n--- labeled spans ---\n{legend}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="+", help="one or more annotated split JSONLs")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--targeted-fraction", type=float, default=0.3)
    parser.add_argument("--mixture-fraction", type=float, default=0.25)
    parser.add_argument("--max-regions", type=int, default=4)
    parser.add_argument("--delimiter-wrap-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    dataset = FragmentDataset(
        args.path,
        fragment_length=args.length,
        samples_per_epoch=args.count,
        targeted_fraction=args.targeted_fraction,
        seed=args.seed,
        mixture_fraction=args.mixture_fraction,
        max_regions=args.max_regions,
        delimiter_wrap_fraction=args.delimiter_wrap_fraction,
    )
    for index in range(len(dataset)):
        if index:
            print("\n" + "=" * 72 + "\n")
        print(describe_sample(dataset, index))


if __name__ == "__main__":
    main()
