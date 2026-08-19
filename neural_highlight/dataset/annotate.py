"""Annotate complete Python files with one normalized label per UTF-8 byte."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable
import re

import tree_sitter_python
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_css
import tree_sitter_go
import tree_sitter_html
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_rust
import tree_sitter_typescript
from tree_sitter import Language, Parser, Query, QueryCursor

from neural_highlight.dataset.normalize import normalize_capture
from neural_highlight.labels import Label, label_name


# The upstream query aims at editor themes and intentionally uses fairly broad
# captures. These supplemental captures preserve distinctions required by our
# compact training ontology and label structural punctuation explicitly.
_SUPPLEMENTAL_QUERY = r"""
(function_definition name: (identifier) @function.definition)
(lambda_parameters (identifier) @variable.parameter)
(parameters (identifier) @variable.parameter)
(attribute attribute: (identifier) @property)
[
  "(" ")" "[" "]" "{" "}" "," ":" ";" "."
] @punctuation
"""
_TYPESCRIPT_QUERY = (
    files("tree_sitter_typescript").joinpath("queries/highlights.scm").read_text(encoding="utf-8")
)

_LANGUAGE_SPECS = {
    "python": (tree_sitter_python.language, tree_sitter_python.HIGHLIGHTS_QUERY + "\n" + _SUPPLEMENTAL_QUERY),
    "javascript": (tree_sitter_javascript.language, tree_sitter_javascript.HIGHLIGHTS_QUERY),
    "typescript": (
        tree_sitter_typescript.language_typescript,
        tree_sitter_javascript.HIGHLIGHTS_QUERY + "\n" + _TYPESCRIPT_QUERY,
    ),
    "html": (tree_sitter_html.language, tree_sitter_html.HIGHLIGHTS_QUERY),
    "css": (tree_sitter_css.language, tree_sitter_css.HIGHLIGHTS_QUERY),
    "rust": (tree_sitter_rust.language, tree_sitter_rust.HIGHLIGHTS_QUERY),
    "c": (tree_sitter_c.language, tree_sitter_c.HIGHLIGHTS_QUERY),
    "c++": (
        tree_sitter_cpp.language,
        tree_sitter_c.HIGHLIGHTS_QUERY + "\n" + tree_sitter_cpp.HIGHLIGHTS_QUERY,
    ),
    "go": (tree_sitter_go.language, tree_sitter_go.HIGHLIGHTS_QUERY),
    "java": (tree_sitter_java.language, tree_sitter_java.HIGHLIGHTS_QUERY),
}
SUPPORTED_LANGUAGES = tuple(_LANGUAGE_SPECS)
_COMPILED: dict[str, tuple[Language, Query]] = {}
_PROSE_WORD = re.compile(rb"[A-Za-z][A-Za-z'-]{1,}")
_COMMENT_DECORATION = re.compile(rb"^\s*(?:\*+|//+|#+)?\s*")
_CODE_PUNCTUATION = frozenset(b"{}[]();=<>`|&")


def _teacher(language: str) -> tuple[Language, Query]:
    language = language.lower()
    if language not in _LANGUAGE_SPECS:
        raise ValueError(f"unsupported language {language!r}; choose from {', '.join(SUPPORTED_LANGUAGES)}")
    if language not in _COMPILED:
        language_factory, query_source = _LANGUAGE_SPECS[language]
        grammar = Language(language_factory())
        _COMPILED[language] = grammar, Query(grammar, query_source)
    return _COMPILED[language]


@dataclass(frozen=True)
class Capture:
    start_byte: int
    end_byte: int
    name: str
    label: Label


@dataclass(frozen=True)
class Annotation:
    source: bytes
    labels: bytes
    captures: tuple[Capture, ...]
    label_mask: bytes | None = None

    def __post_init__(self) -> None:
        if len(self.source) != len(self.labels):
            raise ValueError("source and labels must have identical byte lengths")
        if self.label_mask is not None and len(self.source) != len(self.label_mask):
            raise ValueError("source and label mask must have identical byte lengths")


def _comment_delimiters(source: bytes) -> tuple[int, int]:
    """Return supervised opening/closing delimiter lengths for a comment capture."""
    for opening, closing in (
        (b"<!--", b"-->"),
        (b"/**", b"*/"),
        (b"/*!", b"*/"),
        (b"/*", b"*/"),
        (b"///", b""),
        (b"//!", b""),
        (b"//", b""),
        (b"#", b""),
    ):
        if source.startswith(opening):
            return len(opening), len(closing) if closing and source.endswith(closing) else 0
    return 0, 0


def annotate(
    source: str | bytes,
    language: str,
    *,
    supervise_comment_bodies: bool = False,
) -> Annotation:
    """Parse and annotate a complete source unit, preserving UTF-8 byte offsets."""
    source_bytes = source.encode("utf-8") if isinstance(source, str) else source
    grammar, query = _teacher(language)
    tree = Parser(grammar).parse(source_bytes)
    raw_captures = QueryCursor(query).captures(tree.root_node)
    labels = bytearray([Label.PLAIN]) * len(source_bytes)
    label_mask = bytearray([1]) * len(source_bytes)
    captures: list[Capture] = []

    # The API groups nodes by capture name. Apply broader captures first so a
    # smaller, more specific capture wins when query ranges overlap.
    flattened = [
        Capture(node.start_byte, node.end_byte, name, normalize_capture(name))
        for name, nodes in raw_captures.items()
        for node in nodes
    ]
    flattened.sort(key=lambda item: (-(item.end_byte - item.start_byte), item.start_byte))
    for capture in flattened:
        if capture.label is Label.COMMENT:
            captured = source_bytes[capture.start_byte : capture.end_byte]
            opening_length, closing_length = _comment_delimiters(captured)
            if supervise_comment_bodies:
                labels[capture.start_byte : capture.end_byte] = bytes([Label.COMMENT]) * (
                    capture.end_byte - capture.start_byte
                )
                captures.append(capture)
                continue
            label_mask[capture.start_byte : capture.end_byte] = bytes(
                capture.end_byte - capture.start_byte
            )
            if opening_length:
                opening_end = capture.start_byte + opening_length
                label_mask[capture.start_byte : opening_end] = bytes([1]) * opening_length
                labels[capture.start_byte : opening_end] = bytes([Label.COMMENT]) * opening_length
            if closing_length:
                closing_start = capture.end_byte - closing_length
                label_mask[closing_start : capture.end_byte] = bytes([1]) * closing_length
                labels[closing_start : capture.end_byte] = bytes([Label.COMMENT]) * closing_length
            captures.append(capture)
            continue
        if capture.label is not Label.PLAIN:
            labels[capture.start_byte : capture.end_byte] = bytes([capture.label]) * (
                capture.end_byte - capture.start_byte
            )
        captures.append(capture)

    return Annotation(source_bytes, bytes(labels), tuple(captures), bytes(label_mask))


def _looks_like_prose(line: bytes) -> bool:
    cleaned = _COMMENT_DECORATION.sub(b"", line).strip()
    if not cleaned:
        return True
    words = _PROSE_WORD.findall(cleaned)
    visible = sum(not chr(value).isspace() for value in cleaned)
    if not visible:
        return True
    punctuation = sum(value in _CODE_PUNCTUATION for value in cleaned)
    alpha_ratio = sum(chr(value).isalpha() for value in cleaned) / visible
    marker = cleaned.upper().startswith((b"TODO", b"FIXME", b"NOTE", b"WARNING"))
    return punctuation / visible <= 0.08 and alpha_ratio >= 0.5 and (
        len(words) >= 3 or marker or cleaned.endswith((b".", b"!", b"?", b":"))
    )


def sanitize_comment_code(source: str | bytes, language: str) -> bytes:
    """Remove code-like lines and fenced blocks from teacher-captured comments."""
    source_bytes = source.encode("utf-8") if isinstance(source, str) else source
    annotation = annotate(source_bytes, language)
    comments = [capture for capture in annotation.captures if capture.label is Label.COMMENT]
    if not comments:
        return source_bytes
    output = bytearray()
    cursor = 0
    for capture in sorted(comments, key=lambda item: item.start_byte):
        if capture.start_byte < cursor:
            continue
        output.extend(source_bytes[cursor : capture.start_byte])
        captured = source_bytes[capture.start_byte : capture.end_byte]
        opening_length, closing_length = _comment_delimiters(captured)
        opening = captured[:opening_length]
        closing = captured[len(captured) - closing_length :] if closing_length else b""
        body_end = len(captured) - closing_length if closing_length else len(captured)
        body = captured[opening_length:body_end]
        kept: list[bytes] = []
        fenced = False
        for line in body.splitlines(keepends=True):
            cleaned = _COMMENT_DECORATION.sub(b"", line).strip()
            if cleaned.startswith((b"```", b"~~~")):
                fenced = not fenced
                continue
            if not fenced and _looks_like_prose(line):
                kept.append(line)
        output.extend(opening + b"".join(kept) + closing)
        cursor = capture.end_byte
    output.extend(source_bytes[cursor:])
    return bytes(output)


def annotate_python(source: str | bytes) -> Annotation:
    """Compatibility wrapper for the original Python-only public API."""
    return annotate(source, "python")


def iter_spans(annotation: Annotation) -> Iterable[tuple[int, int, Label, bytes]]:
    """Coalesce adjacent byte labels into spans, including plain whitespace."""
    if not annotation.source:
        return
    start = 0
    current = Label(annotation.labels[0])
    for index, raw_label in enumerate(annotation.labels[1:], 1):
        label = Label(raw_label)
        if label != current:
            yield start, index, current, annotation.source[start:index]
            start, current = index, label
    yield start, len(annotation.source), current, annotation.source[start:]


_ANSI = {
    Label.KEYWORD: "\033[95m",
    Label.IDENTIFIER: "\033[37m",
    Label.FUNCTION: "\033[94m",
    Label.FUNCTION_DEFINITION: "\033[1;94m",
    Label.PARAMETER: "\033[96m",
    Label.TYPE: "\033[96m",
    Label.PROPERTY: "\033[36m",
    Label.STRING: "\033[92m",
    Label.COMMENT: "\033[90m",
    Label.NUMBER: "\033[93m",
    Label.OPERATOR: "\033[91m",
    Label.PUNCTUATION: "\033[97m",
    Label.CONSTANT: "\033[93m",
}
_RESET = "\033[0m"


def colored(annotation: Annotation) -> str:
    return "".join(
        f"{_ANSI.get(label, '')}{text.decode('utf-8', errors='replace')}{_RESET if label in _ANSI else ''}"
        for _, _, label, text in iter_spans(annotation)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="complete Python file to annotate")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="python")
    parser.add_argument("--spans", action="store_true", help="print byte ranges and labels")
    args = parser.parse_args(argv)
    annotation = annotate(args.path.read_bytes(), args.language)
    if args.spans:
        for start, end, label, text in iter_spans(annotation):
            print(f"{start:6}:{end:<6} {label_name(label):20} {text!r}")
    else:
        print(colored(annotation), end="")


if __name__ == "__main__":
    main()
