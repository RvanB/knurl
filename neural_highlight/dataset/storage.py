"""Portable JSONL storage for fully annotated source files."""

from __future__ import annotations

import base64
import json
import mmap
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


_LANGUAGE_FIELD = re.compile(br'"language":"([^"]+)"')


@dataclass(frozen=True)
class StoredFile:
    repository: str
    language: str
    path: str
    source: bytes
    labels: bytes
    region_labels: bytes | None = None

    def to_json(self) -> str:
        if len(self.source) != len(self.labels):
            raise ValueError("source and labels must have equal lengths")
        if self.region_labels is not None and len(self.source) != len(self.region_labels):
            raise ValueError("source and region labels must have equal lengths")
        value = {
                "repository": self.repository,
                "language": self.language,
                "path": self.path,
                "source_b64": base64.b64encode(self.source).decode("ascii"),
                "labels_b64": base64.b64encode(self.labels).decode("ascii"),
            }
        if self.region_labels is not None:
            value["region_labels_b64"] = base64.b64encode(self.region_labels).decode("ascii")
        return json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_json(cls, line: str) -> "StoredFile":
        value = json.loads(line)
        return cls(
            repository=value["repository"],
            language=value["language"],
            path=value["path"],
            source=base64.b64decode(value["source_b64"]),
            labels=base64.b64decode(value["labels_b64"]),
            region_labels=(
                base64.b64decode(value["region_labels_b64"])
                if "region_labels_b64" in value
                else None
            ),
        )


def write_record(stream: TextIO, record: StoredFile) -> None:
    stream.write(record.to_json() + "\n")


def read_records(path: Path) -> Iterator[StoredFile]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield StoredFile.from_json(line)


@dataclass(frozen=True)
class RecordRef:
    file_index: int
    offset: int
    length: int
    language: str


class IndexedJsonlStore:
    """Random-access JSONL reader retaining offsets rather than decoded files."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = tuple(paths)
        refs: list[RecordRef] = []
        for file_index, path in enumerate(self.paths):
            with path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    match = _LANGUAGE_FIELD.search(line[:2048])
                    if match is None:
                        raise ValueError(f"record at {path}:{offset} has no language field")
                    refs.append(
                        RecordRef(
                            file_index, offset, len(line), match.group(1).decode("utf-8")
                        )
                    )
        self.refs = tuple(refs)
        self._files: list[object | None] = [None] * len(self.paths)
        self._maps: list[mmap.mmap | None] = [None] * len(self.paths)

    def __len__(self) -> int:
        return len(self.refs)

    def language(self, index: int) -> str:
        return self.refs[index].language

    def __getitem__(self, index: int) -> StoredFile:
        ref = self.refs[index]
        mapping = self._maps[ref.file_index]
        if mapping is None:
            stream = self.paths[ref.file_index].open("rb")
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            self._files[ref.file_index] = stream
            self._maps[ref.file_index] = mapping
        line = mapping[ref.offset : ref.offset + ref.length].decode("utf-8")
        return StoredFile.from_json(line)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_files"] = [None] * len(self.paths)
        state["_maps"] = [None] * len(self.paths)
        return state

    def close(self) -> None:
        for mapping in self._maps:
            if mapping is not None:
                mapping.close()
        for stream in self._files:
            if stream is not None:
                stream.close()  # type: ignore[union-attr]
        self._maps = [None] * len(self.paths)
        self._files = [None] * len(self.paths)
