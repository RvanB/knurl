"""Conservative prose extraction and prose/code training augmentation."""

from __future__ import annotations

import random
import re
from neural_highlight.dataset.storage import IndexedJsonlStore, StoredFile


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")
_DECORATION = re.compile(r"^\s*(?:\*+|//+|#+|<!--|-->)?\s*")
_CODE_PUNCTUATION = frozenset("{}[]();=<>`|&")


def prose_lines(record: StoredFile) -> tuple[bytes, ...]:
    """Return high-confidence natural-language lines from masked comment bodies."""
    if record.label_mask is None:
        return ()
    candidates: list[bytes] = []
    start = 0
    while start < len(record.source):
        if record.label_mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(record.source) and not record.label_mask[end]:
            end += 1
        text = record.source[start:end].decode("utf-8", errors="ignore")
        for raw_line in text.splitlines():
            line = _DECORATION.sub("", raw_line).strip()
            words = _WORD.findall(line)
            visible = sum(not char.isspace() for char in line)
            punctuation = sum(char in _CODE_PUNCTUATION for char in line)
            if (
                len(words) >= 4
                and visible >= 16
                and sum(char.isalpha() for char in line) / visible >= 0.55
                and punctuation / visible <= 0.08
            ):
                candidates.append(line.encode("utf-8"))
        start = end
    return tuple(candidates)


def sample_prose(
    store: IndexedJsonlStore,
    rng: random.Random,
    max_bytes: int,
    attempts: int = 32,
) -> bytes | None:
    """Lazily find a prose line without retaining decoded corpus contents."""
    for _ in range(attempts):
        lines = prose_lines(store[rng.randrange(len(store))])
        if lines:
            value = rng.choice(lines)
            if len(value) <= max_bytes:
                return value
            start = rng.randrange(len(value) - max_bytes + 1)
            return value[start : start + max_bytes]
    return None
