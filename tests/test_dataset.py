from pathlib import Path

from neural_highlight.dataset.download import take_usable_files
from neural_highlight.dataset.split import repository_split
from neural_highlight.dataset.storage import StoredFile, read_records, write_record


def test_repository_split_is_stable_and_repository_scoped() -> None:
    assert repository_split("example/project", 42) == repository_split("example/project", 42)
    assert repository_split("example/project", 42) in {"train", "validation", "test"}


def test_storage_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    expected = StoredFile("repo", "python", "café.py", "é".encode(), bytes([8, 8]))
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
