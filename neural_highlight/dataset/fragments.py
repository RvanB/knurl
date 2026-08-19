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
from neural_highlight.dataset.storage import StoredFile, read_records
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
    ) -> None:
        if fragment_length <= 0:
            raise ValueError("fragment_length must be positive")
        if not 0.0 <= targeted_fraction <= 1.0:
            raise ValueError("targeted_fraction must be between zero and one")
        if not 0.0 <= mixture_fraction <= 1.0:
            raise ValueError("mixture_fraction must be between zero and one")
        if max_regions < 2:
            raise ValueError("max_regions must be at least two")
        paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(item) for item in path]
        self.paths = tuple(paths)
        self.records = tuple(record for item in paths for record in read_records(item))
        if not self.records:
            raise ValueError(f"no records found in {paths}")
        self.fragment_length = fragment_length
        self.samples_per_epoch = samples_per_epoch or len(self.records)
        self.targeted_fraction = targeted_fraction
        self.seed = seed
        self.mixture_fraction = mixture_fraction
        self.max_regions = max_regions
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
        self._interesting = tuple(
            tuple(index for index, label in enumerate(record.labels) if label != Label.PLAIN)
            for record in self.records
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(epoch)

    def _rng(self, index: int) -> random.Random:
        return random.Random((self.seed << 64) ^ (int(self._epoch) << 32) ^ index)

    def _crop_start(self, record_index: int, rng: random.Random) -> int:
        record = self.records[record_index]
        largest_start = max(0, len(record.source) - self.fragment_length)
        interesting = self._interesting[record_index]
        if interesting and rng.random() < self.targeted_fraction:
            center = rng.choice(interesting)
            jitter = rng.randrange(-self.fragment_length // 4, self.fragment_length // 4 + 1)
            return min(largest_start, max(0, center - self.fragment_length // 2 + jitter))
        return rng.randrange(largest_start + 1)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = self._rng(index)
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
        record_index = rng.randrange(len(self.records))
        record = self.records[record_index]
        start = self._crop_start(record_index, rng)
        source = record.source[start : start + self.fragment_length]
        labels = record.labels[start : start + self.fragment_length]
        language_id = LANGUAGE_IDS.get(record.language.lower(), LANGUAGE_IDS["unknown"])
        return self._pack(
            source, labels, self._region_bytes(record, start, len(source)),
            language_id, record_index, start,
        )

    def _mixed_sample(self, rng: random.Random) -> dict[str, Tensor]:
        region_count = rng.randint(2, min(self.max_regions, self.fragment_length))
        boundaries = sorted(rng.sample(range(1, self.fragment_length), region_count - 1))
        lengths = [end - start for start, end in zip((0, *boundaries), (*boundaries, self.fragment_length))]
        sources: list[bytes] = []
        labels: list[bytes] = []
        regions: list[bytes] = []
        first_record_index = 0
        previous_language: str | None = None
        for region_index, length in enumerate(lengths):
            candidates = [
                index for index, candidate in enumerate(self.records)
                if candidate.language != previous_language
            ] or list(range(len(self.records)))
            record_index = rng.choice(candidates)
            if region_index == 0:
                first_record_index = record_index
            record = self.records[record_index]
            previous_language = record.language
            largest_start = max(0, len(record.source) - length)
            start = rng.randrange(largest_start + 1)
            part_source = record.source[start : start + length]
            sources.append(part_source)
            labels.append(record.labels[start : start + length])
            regions.append(self._region_bytes(record, start, len(part_source)))
        return self._pack(
            b"".join(sources), b"".join(labels), b"".join(regions),
            LANGUAGE_IDS["unknown"], first_record_index, -1,
        )

    def source_for(self, sample: dict[str, Tensor]) -> bytes:
        length = int(sample["source_length"])
        return bytes(sample["input_ids"][:length].tolist())


def describe_sample(dataset: FragmentDataset, index: int) -> str:
    sample = dataset[index]
    length = int(sample["source_length"])
    source = dataset.source_for(sample)
    labels = bytes(sample["labels"][:length].tolist())
    region_labels = sample["region_labels"][:length].tolist()
    annotation = Annotation(source, labels, ())
    record = dataset.records[int(sample["file_index"])]
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
    args = parser.parse_args(argv)
    dataset = FragmentDataset(
        args.path,
        fragment_length=args.length,
        samples_per_epoch=args.count,
        targeted_fraction=args.targeted_fraction,
        seed=args.seed,
        mixture_fraction=args.mixture_fraction,
        max_regions=args.max_regions,
    )
    for index in range(len(dataset)):
        if index:
            print("\n" + "=" * 72 + "\n")
        print(describe_sample(dataset, index))


if __name__ == "__main__":
    main()
