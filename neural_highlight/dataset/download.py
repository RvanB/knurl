"""Stream source records from The Stack Smol without downloading all languages."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import hashlib
from typing import Any


STACK_SMOL_DATASET = "bigcode/the-stack-smol"
STACK_SMOL_XS_DATASET = "bigcode/the-stack-smol-xs"
SMOL_XS_ONLY_LANGUAGES = frozenset(("glsl",))
_EXTENSIONS = {
    "python": "py", "javascript": "js", "typescript": "ts", "html": "html",
    "css": "css", "rust": "rs", "c": "c", "c++": "cpp", "go": "go", "java": "java",
    "shell": "sh", "sql": "sql", "markdown": "md", "lua": "lua", "ruby": "rb",
    "glsl": "glsl",
}


def stream_stack_smol(language: str = "python") -> Iterator[Mapping[str, Any]]:
    # Imported lazily so annotation/tests do not pay PyArrow's import cost.
    from datasets import load_dataset

    dataset = load_dataset(
        STACK_SMOL_DATASET,
        data_dir=f"data/{language.lower()}",
        split="train",
        streaming=True,
    )
    yield from dataset


def stream_stack_smol_xs(language: str = "python") -> Iterator[Mapping[str, Any]]:
    """Read the public XS JSONL fallback (which lacks repository metadata)."""
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=(
            f"https://huggingface.co/datasets/{STACK_SMOL_XS_DATASET}/resolve/"
            "refs%2Fconvert%2Fparquet/"
            f"{language.lower()}/train/0000.parquet"
        ),
        split="train",
        streaming=True,
    )
    for index, source_record in enumerate(dataset):
        record = dict(source_record)
        digest = hashlib.sha256(record["content"].encode("utf-8")).hexdigest()
        # XS omits repository and path. Content-derived IDs are explicit and
        # stable, but cannot provide leakage protection at repository level.
        record["repository_name"] = f"xs-content-{digest}"
        extension = record.get("ext") or _EXTENSIONS.get(language, "txt")
        record["path"] = f"{index:06d}.{extension}"
        yield record


def take_usable_files(
    records: Iterable[Mapping[str, Any]],
    max_files: int,
    max_bytes: int,
) -> Iterator[Mapping[str, Any]]:
    accepted = 0
    for record in records:
        content = record.get("content")
        repository = record.get("repository_name")
        path = record.get("path")
        if not isinstance(content, str) or not content or not repository or not path:
            continue
        if len(content.encode("utf-8")) > max_bytes:
            continue
        yield record
        accepted += 1
        if accepted >= max_files:
            return
