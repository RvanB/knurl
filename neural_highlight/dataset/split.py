"""Stable repository-level train/validation/test assignment."""

from __future__ import annotations

import hashlib


def repository_split(repository: str, seed: int = 0) -> str:
    digest = hashlib.sha256(f"{seed}\0{repository}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"

