from pathlib import Path

import torch
import pytest

from neural_highlight.dataset.comment_wrapping import wrap_code_as_comment
from neural_highlight.dataset.fragments import IGNORE_LABEL_ID, PAD_BYTE_ID, FragmentDataset
from neural_highlight.dataset.storage import StoredFile, write_record
from neural_highlight.dataset.prose import prose_lines
from neural_highlight.labels import Label


def make_dataset(path: Path) -> None:
    records = [
        StoredFile(
            "r1", "python", "short.py", b"x=1",
            bytes([Label.IDENTIFIER, Label.OPERATOR, Label.NUMBER]),
        ),
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
    assert dataset.epoch_is_shared == dataset._epoch.is_shared()
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
        mixture_fraction=1.0, max_regions=2, delimiter_wrap_fraction=0,
    )
    for sample in dataset:
        valid = sample["region_labels"][sample["region_labels"] != IGNORE_LABEL_ID]
        assert len(torch.unique(valid)) == 2
        assert int(sample["language_id"]) == 0


def test_dataset_combines_multiple_split_files(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    make_dataset(first)
    make_dataset(second)
    dataset = FragmentDataset([first, second], fragment_length=4)
    assert dataset.record_count == 4


def test_supervision_mask_becomes_ignored_targets(tmp_path: Path) -> None:
    path = tmp_path / "masked.jsonl"
    record = StoredFile(
        "r", "rust", "x.rs", b"/*code*/", bytes([Label.COMMENT]) * 8,
        label_mask=bytes([1, 1, 0, 0, 0, 0, 1, 1]),
    )
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, record)
    sample = FragmentDataset(
        path, fragment_length=8, samples_per_epoch=1, mixture_fraction=0
    )[0]
    assert sample["labels"].tolist() == [
        Label.COMMENT, Label.COMMENT, -100, -100, -100, -100,
        Label.COMMENT, Label.COMMENT,
    ]
    assert sample["region_labels"].tolist()[2:6] == [-100] * 4


def test_delimiter_wrapping_preserves_code_targets(tmp_path: Path) -> None:
    path = tmp_path / "code.jsonl"
    record = StoredFile("r", "java", "X.java", b"return value", bytes([1] * 6 + [0] + [2] * 5))
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, record)
    sample = FragmentDataset(
        path, fragment_length=16, samples_per_epoch=1,
        mixture_fraction=0, delimiter_wrap_fraction=1,
    )[0]
    source = bytes(sample["input_ids"].tolist())
    assert b"return" in source
    start = source.index(b"return")
    assert sample["labels"][start : start + 6].tolist() == [Label.KEYWORD] * 6


@pytest.mark.parametrize(
    ("style", "opening", "line_prefix", "closing"),
    (
        ("c_multiline", b"/*\n", b"", b"\n*/"),
        ("c_starred", b"/*\n * ", b" * ", b"\n */"),
        ("html_multiline", b"<!--\n", b"", b"\n-->"),
        ("slash_lines", b"// ", b"// ", b""),
        ("hash_lines", b"# ", b"# ", b""),
    ),
)
def test_multiline_comment_wrappers_preserve_code_labels(
    style: str, opening: bytes, line_prefix: bytes, closing: bytes,
) -> None:
    source = b"int x = 1;\nreturn x;"
    labels = bytes(range(len(source)))
    regions = bytes([11]) * len(source)
    wrapped = wrap_code_as_comment(
        source, labels, regions, bytes([1]) * len(source), 128, style=style,
    )

    assert wrapped.source.startswith(opening)
    assert wrapped.source.endswith(closing)
    if line_prefix:
        assert b"\n" + line_prefix + b"return" in wrapped.source
    first, second = source.splitlines(keepends=True)
    first_start = wrapped.source.index(first)
    second_start = wrapped.source.index(second)
    assert wrapped.labels[first_start : first_start + len(first)] == labels[: len(first)]
    assert wrapped.labels[second_start : second_start + len(second)] == labels[len(first) :]
    decoration_positions = [
        index for index, label in enumerate(wrapped.labels) if label == Label.COMMENT
    ]
    assert decoration_positions
    assert all(wrapped.mask[index] for index in decoration_positions)


def test_prose_extraction_and_augmentation_supervise_both_classes(tmp_path: Path) -> None:
    path = tmp_path / "prose.jsonl"
    prose = b"This function returns the configured value."
    source = b"/* " + prose + b" */\nreturn value"
    labels = bytes([Label.PLAIN]) * len(source)
    labels = labels[:-12] + bytes([Label.KEYWORD]) * 6 + bytes([Label.PLAIN]) + bytes([Label.IDENTIFIER]) * 5
    mask = bytearray([1]) * len(source)
    mask[3 : 3 + len(prose)] = bytes(len(prose))
    record = StoredFile("r", "python", "x.py", source, labels, label_mask=bytes(mask))
    assert prose_lines(record) == (prose,)
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, record)
    sample = FragmentDataset(
        path, fragment_length=96, samples_per_epoch=1, mixture_fraction=0,
        delimiter_wrap_fraction=0, prose_code_fraction=1,
    )[0]
    rendered = bytes(sample["input_ids"][: int(sample["source_length"])] .tolist())
    prose_start = rendered.index(prose)
    code_start = rendered.rindex(b"return")
    assert sample["labels"][prose_start : prose_start + len(prose)].tolist() == [Label.COMMENT] * len(prose)
    assert sample["labels"][code_start : code_start + 6].tolist() == [Label.KEYWORD] * 6
