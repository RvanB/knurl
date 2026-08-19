"""PyTorch dataset for fixed-length fragments of fully annotated files."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from neural_highlight.dataset.annotate import Annotation, colored, iter_spans
from neural_highlight.dataset.storage import StoredFile, read_records
from neural_highlight.labels import Label, label_name


PAD_BYTE_ID = 256
IGNORE_LABEL_ID = -100
LANGUAGE_IDS = {"unknown": 0, "python": 1}


@dataclass(frozen=True)
class Fragment:
    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    language_id: Tensor
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
        path: str | Path,
        fragment_length: int = 256,
        samples_per_epoch: int | None = None,
        targeted_fraction: float = 0.3,
        seed: int = 0,
    ) -> None:
        if fragment_length <= 0:
            raise ValueError("fragment_length must be positive")
        if not 0.0 <= targeted_fraction <= 1.0:
            raise ValueError("targeted_fraction must be between zero and one")
        self.records = tuple(read_records(Path(path)))
        if not self.records:
            raise ValueError(f"no records found in {path}")
        self.fragment_length = fragment_length
        self.samples_per_epoch = samples_per_epoch or len(self.records)
        self.targeted_fraction = targeted_fraction
        self.seed = seed
        self.epoch = 0
        self._interesting = tuple(
            tuple(index for index, label in enumerate(record.labels) if label != Label.PLAIN)
            for record in self.records
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, index: int) -> random.Random:
        return random.Random((self.seed << 64) ^ (self.epoch << 32) ^ index)

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
        record_index = rng.randrange(len(self.records))
        record = self.records[record_index]
        start = self._crop_start(record_index, rng)
        source = record.source[start : start + self.fragment_length]
        labels = record.labels[start : start + self.fragment_length]
        source_length = len(source)
        padding = self.fragment_length - source_length
        language_id = LANGUAGE_IDS.get(record.language, LANGUAGE_IDS["unknown"])
        return {
            "input_ids": torch.tensor(list(source) + [PAD_BYTE_ID] * padding, dtype=torch.long),
            "labels": torch.tensor(list(labels) + [IGNORE_LABEL_ID] * padding, dtype=torch.long),
            "attention_mask": torch.tensor([True] * source_length + [False] * padding),
            "language_id": torch.tensor(language_id, dtype=torch.long),
            "source_length": torch.tensor(source_length, dtype=torch.long),
            "file_index": torch.tensor(record_index, dtype=torch.long),
            "start_byte": torch.tensor(start, dtype=torch.long),
        }

    def source_for(self, sample: dict[str, Tensor]) -> bytes:
        length = int(sample["source_length"])
        return bytes(sample["input_ids"][:length].tolist())


def describe_sample(dataset: FragmentDataset, index: int) -> str:
    sample = dataset[index]
    length = int(sample["source_length"])
    source = dataset.source_for(sample)
    labels = bytes(sample["labels"][:length].tolist())
    annotation = Annotation(source, labels, ())
    record = dataset.records[int(sample["file_index"])]
    header = (
        f"{record.repository}/{record.path} bytes {int(sample['start_byte'])}:"
        f"{int(sample['start_byte']) + length} ({length}/{dataset.fragment_length})"
    )
    legend = "\n".join(
        f"{start:4}:{end:<4} {label_name(label):20} {text!r}"
        for start, end, label, text in iter_spans(annotation)
        if label is not Label.PLAIN
    )
    return f"{header}\n{colored(annotation)}\n--- labeled spans ---\n{legend}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="an annotated split JSONL")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--targeted-fraction", type=float, default=0.3)
    args = parser.parse_args(argv)
    dataset = FragmentDataset(
        args.path,
        fragment_length=args.length,
        samples_per_epoch=args.count,
        targeted_fraction=args.targeted_fraction,
        seed=args.seed,
    )
    for index in range(len(dataset)):
        if index:
            print("\n" + "=" * 72 + "\n")
        print(describe_sample(dataset, index))


if __name__ == "__main__":
    main()

