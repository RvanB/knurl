"""Synthetic multiline string wrappers with syntax-labeled embedded code."""

from __future__ import annotations

import random
from dataclasses import dataclass

from neural_highlight.labels import Label


@dataclass(frozen=True)
class WrappedStringCode:
    source: bytes
    labels: bytes
    regions: bytes
    mask: bytes


_STYLES_BY_LANGUAGE = {
    "python": ("python_double", "python_single"),
    "javascript": ("backtick",),
    "typescript": ("backtick",),
    "java": ("java_text_block",),
    "rust": ("rust_raw",),
    "c++": ("cpp_raw",),
    "go": ("backtick",),
}


def string_wrapper_languages() -> tuple[str, ...]:
    return tuple(_STYLES_BY_LANGUAGE)


def random_string_wrapper_style(language: str, rng: random.Random) -> str:
    return rng.choice(_STYLES_BY_LANGUAGE[language])


def wrap_code_as_string(
    source: bytes,
    labels: bytes,
    regions: bytes,
    mask: bytes,
    max_length: int,
    *,
    style: str,
    host_region: int,
) -> WrappedStringCode:
    """Wrap as much code as fits and label only literal delimiters as string."""
    if not (len(source) == len(labels) == len(regions) == len(mask)):
        raise ValueError("source and target arrays must have equal lengths")
    delimiters = {
        "python_double": (b'"""\n', b'\n"""'),
        "python_single": (b"'''\n", b"\n'''"),
        "backtick": (b"`\n", b"\n`"),
        "java_text_block": (b'"""\n', b'\n"""'),
        "rust_raw": (b'r#"\n', b'\n"#'),
        "cpp_raw": (b'R"CODE(\n', b'\n)CODE"'),
    }
    if style not in delimiters:
        raise ValueError(f"unknown string wrapper style {style!r}")
    opening, closing = delimiters[style]
    available = max_length - len(opening) - len(closing)
    if available < 1:
        raise ValueError("max_length is too short for string wrapper")
    payload = source[:available]
    # Do not synthesize a literal that closes before the intended boundary.
    early_closing = payload.find(closing.strip())
    if early_closing >= 0:
        payload = payload[:early_closing]
    payload_length = len(payload)
    string = bytes([Label.STRING])
    return WrappedStringCode(
        opening + payload + closing,
        string * len(opening) + labels[:payload_length] + string * len(closing),
        bytes([host_region]) * len(opening)
        + regions[:payload_length]
        + bytes([host_region]) * len(closing),
        bytes([1]) * len(opening) + mask[:payload_length] + bytes([1]) * len(closing),
    )
