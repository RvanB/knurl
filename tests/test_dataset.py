from pathlib import Path

import pytest

from neural_highlight.dataset.download import take_usable_files
from neural_highlight.dataset.split import repository_split
from neural_highlight.dataset.storage import IndexedJsonlStore, StoredFile, read_records, write_record
from neural_highlight.labels import LABEL_SCHEMA_VERSION, Label


def test_repository_split_is_stable_and_repository_scoped() -> None:
    assert repository_split("example/project", 42) == repository_split("example/project", 42)
    assert repository_split("example/project", 42) in {"train", "validation", "test"}


def test_storage_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    expected = StoredFile("repo", "python", "café.py", "é".encode(), bytes([8, 8]))
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, expected)
    assert list(read_records(path)) == [expected]


def test_storage_round_trip_with_region_labels(tmp_path: Path) -> None:
    path = tmp_path / "regions.jsonl"
    expected = StoredFile(
        "repo", "python", "x.py", b"x=1",
        bytes([Label.IDENTIFIER, Label.OPERATOR, Label.NUMBER]), bytes([2, 2, 2]),
    )
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, expected)
    assert list(read_records(path)) == [expected]


def test_storage_round_trip_with_supervision_mask(tmp_path: Path) -> None:
    path = tmp_path / "masked.jsonl"
    expected = StoredFile(
        "repo", "rust", "x.rs", b"/*x*/", bytes([Label.COMMENT]) * 5,
        label_mask=bytes([1, 1, 0, 1, 1]),
    )
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, expected)
    assert list(read_records(path)) == [expected]


def test_filter_caps_accepted_files_not_input_rows() -> None:
    rows = [
        {"content": "", "repository_name": "r", "path": "empty.py"},
        {"content": "x" * 20, "repository_name": "r", "path": "large.py"},
        {"content": "x = 1", "repository_name": "r1", "path": "a.py"},
        {"content": "y = 2", "repository_name": "r2", "path": "b.py"},
    ]
    result = list(take_usable_files(rows, max_files=1, max_bytes=10))
    assert [row["path"] for row in result] == ["a.py"]


def test_indexed_store_reads_records_lazily(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        StoredFile("r1", "python", "a.py", b"x", bytes([2])),
        StoredFile("r2", "javascript", "b.js", b"y", bytes([2])),
    ]
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            write_record(stream, record)
    store = IndexedJsonlStore([path])
    assert len(store) == 2
    assert store.language(1) == "javascript"
    assert all(mapping is None for mapping in store._maps)
    assert store[0] == records[0]
    assert store._maps[0] is not None
    store.close()


def test_indexed_store_rejects_legacy_label_schema(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    line = StoredFile("r", "python", "x.py", b"x", bytes([Label.IDENTIFIER])).to_json()
    line = line.replace(f'"label_schema_version":{LABEL_SCHEMA_VERSION},', "")
    path.write_text(line + "\n", encoding="utf-8")
    store = IndexedJsonlStore([path])
    with pytest.raises(ValueError, match="regenerate the annotated dataset"):
        store.require_current_label_schema()
