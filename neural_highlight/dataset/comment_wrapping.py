"""Synthetic comment wrappers that preserve labels on embedded code bytes."""

from __future__ import annotations

import random
from dataclasses import dataclass

from neural_highlight.labels import Label


@dataclass(frozen=True)
class WrappedCode:
    source: bytes
    labels: bytes
    regions: bytes
    mask: bytes


_STYLES_BY_LANGUAGE = {
    "python": ("hash_lines",),
    "html": ("html_block", "html_multiline"),
    "css": ("c_block", "c_multiline", "c_starred"),
    "javascript": ("c_block", "c_multiline", "c_starred", "slash_lines"),
    "typescript": ("c_block", "c_multiline", "c_starred", "slash_lines"),
    "rust": ("c_block", "c_multiline", "c_starred", "slash_lines"),
    "c": ("c_block", "c_multiline", "c_starred", "slash_lines"),
    "c++": ("c_block", "c_multiline", "c_starred", "slash_lines"),
    "go": ("c_block", "c_multiline", "c_starred", "slash_lines"),
    "java": ("c_block", "c_multiline", "c_starred", "slash_lines"),
}


def comment_wrapper_styles(language: str) -> tuple[str, ...]:
    """Return syntactically relevant wrappers for a supported language."""
    return _STYLES_BY_LANGUAGE.get(language.lower(), ("c_block", "c_multiline"))


def wrap_code_as_comment(
    source: bytes,
    labels: bytes,
    regions: bytes,
    mask: bytes,
    max_length: int,
    *,
    style: str,
) -> WrappedCode:
    """Wrap as much code as fits, labeling only inserted decoration as comment."""
    if not (len(source) == len(labels) == len(regions) == len(mask)):
        raise ValueError("source and target arrays must have equal lengths")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    if style == "c_block":
        opening, line_prefix, closing = b"/*", b"", b"*/"
    elif style == "c_multiline":
        opening, line_prefix, closing = b"/*\n", b"", b"\n*/"
    elif style == "c_starred":
        opening, line_prefix, closing = b"/*\n", b" * ", b"\n */"
    elif style == "html_block":
        opening, line_prefix, closing = b"<!--", b"", b"-->"
    elif style == "html_multiline":
        opening, line_prefix, closing = b"<!--\n", b"", b"\n-->"
    elif style == "slash_lines":
        opening, line_prefix, closing = b"", b"// ", b""
    elif style == "hash_lines":
        opening, line_prefix, closing = b"", b"# ", b""
    else:
        raise ValueError(f"unknown comment wrapper style {style!r}")

    minimum = len(opening) + len(line_prefix) + len(closing)
    if minimum > max_length:
        raise ValueError("max_length is too short for comment wrapper")

    comment = int(Label.COMMENT)
    output_source = bytearray(opening + line_prefix)
    output_labels = bytearray([comment]) * len(output_source)
    output_regions = bytearray([regions[0] if regions else 0]) * len(output_source)
    output_mask = bytearray([1]) * len(output_source)

    for index, value in enumerate(source):
        decoration = line_prefix if value == ord("\n") and index + 1 < len(source) else b""
        if len(output_source) + 1 + len(decoration) + len(closing) > max_length:
            break
        output_source.append(value)
        output_labels.append(labels[index])
        output_regions.append(regions[index])
        output_mask.append(mask[index])
        if decoration:
            output_source.extend(decoration)
            output_labels.extend(bytes([comment]) * len(decoration))
            output_regions.extend(bytes([regions[index]]) * len(decoration))
            output_mask.extend(bytes([1]) * len(decoration))

    output_source.extend(closing)
    output_labels.extend(bytes([comment]) * len(closing))
    closing_region = output_regions[-1] if output_regions else (regions[0] if regions else 0)
    output_regions.extend(bytes([closing_region]) * len(closing))
    output_mask.extend(bytes([1]) * len(closing))
    return WrappedCode(
        bytes(output_source), bytes(output_labels), bytes(output_regions), bytes(output_mask)
    )


def random_comment_wrapper_style(language: str, rng: random.Random) -> str:
    return rng.choice(comment_wrapper_styles(language))
