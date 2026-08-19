from pathlib import Path

import torch

from neural_highlight.dataset.fragments import IGNORE_LABEL_ID, PAD_BYTE_ID, FragmentDataset
from neural_highlight.dataset.storage import StoredFile, write_record
from neural_highlight.labels import Label


def make_dataset(path: Path) -> None:
    records = [
        StoredFile("r1", "python", "short.py", b"x=1", bytes([2, 11, 10])),
        StoredFile(
            "r2",
            "javascript",
            "long.js",
            b"// comment\nvalue",
            bytes([Label.COMMENT] * 10 + [Label.PLAIN] + [Label.IDENTIFIER] * 5),
        ),
    ]
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            write_record(stream, record)


def test_fragment_shapes_padding_and_alignment(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    make_dataset(path)
    dataset = FragmentDataset(path, fragment_length=20, samples_per_epoch=8, seed=4)
    for sample in dataset:
        assert sample["input_ids"].shape == torch.Size([20])
        assert sample["labels"].shape == torch.Size([20])
        length = int(sample["source_length"])
        assert torch.all(sample["input_ids"][length:] == PAD_BYTE_ID)
        assert torch.all(sample["labels"][length:] == IGNORE_LABEL_ID)
        assert torch.all(sample["attention_mask"][:length])
        assert not torch.any(sample["attention_mask"][length:])


def test_sampling_is_deterministic_per_epoch(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    make_dataset(path)
    dataset = FragmentDataset(path, fragment_length=5, samples_per_epoch=20, seed=9)
    first = [dataset[index]["input_ids"].tolist() for index in range(len(dataset))]
    assert first == [dataset[index]["input_ids"].tolist() for index in range(len(dataset))]
    dataset.set_epoch(1)
    assert first != [dataset[index]["input_ids"].tolist() for index in range(len(dataset))]


def test_targeted_sampling_contains_non_plain_label(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    make_dataset(path)
    dataset = FragmentDataset(path, fragment_length=5, samples_per_epoch=20, targeted_fraction=1.0)
    for sample in dataset:
        valid = sample["labels"][sample["labels"] != IGNORE_LABEL_ID]
        assert torch.any(valid != Label.PLAIN)


def test_synthetic_mixture_has_per_byte_language_switches(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    make_dataset(path)
    dataset = FragmentDataset(
        path, fragment_length=6, samples_per_epoch=10,
        mixture_fraction=1.0, max_regions=2,
    )
    for sample in dataset:
        valid = sample["region_labels"][sample["region_labels"] != IGNORE_LABEL_ID]
        assert len(torch.unique(valid)) == 2
        assert int(sample["language_id"]) == 0
