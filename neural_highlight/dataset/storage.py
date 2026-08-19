"""Portable JSONL storage for fully annotated source files."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


@dataclass(frozen=True)
class StoredFile:
    repository: str
    language: str
    path: str
    source: bytes
    labels: bytes

    def to_json(self) -> str:
        if len(self.source) != len(self.labels):
            raise ValueError("source and labels must have equal lengths")
        return json.dumps(
            {
                "repository": self.repository,
                "language": self.language,
                "path": self.path,
                "source_b64": base64.b64encode(self.source).decode("ascii"),
                "labels_b64": base64.b64encode(self.labels).decode("ascii"),
            },
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
        )


def write_record(stream: TextIO, record: StoredFile) -> None:
    stream.write(record.to_json() + "\n")


def read_records(path: Path) -> Iterator[StoredFile]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield StoredFile.from_json(line)

