"""Annotate complete Python files with one normalized label per UTF-8 byte."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor

from neural_highlight.dataset.normalize import normalize_capture
from neural_highlight.labels import Label, label_name


PYTHON_LANGUAGE = Language(tree_sitter_python.language())

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
PYTHON_QUERY = Query(
    PYTHON_LANGUAGE, tree_sitter_python.HIGHLIGHTS_QUERY + "\n" + _SUPPLEMENTAL_QUERY
)


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

    def __post_init__(self) -> None:
        if len(self.source) != len(self.labels):
            raise ValueError("source and labels must have identical byte lengths")


def annotate_python(source: str | bytes) -> Annotation:
    """Parse and annotate a complete source unit, preserving UTF-8 byte offsets."""
    source_bytes = source.encode("utf-8") if isinstance(source, str) else source
    tree = Parser(PYTHON_LANGUAGE).parse(source_bytes)
    raw_captures = QueryCursor(PYTHON_QUERY).captures(tree.root_node)
    labels = bytearray([Label.PLAIN]) * len(source_bytes)
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
        if capture.label is not Label.PLAIN:
            labels[capture.start_byte : capture.end_byte] = bytes([capture.label]) * (
                capture.end_byte - capture.start_byte
            )
        captures.append(capture)

    return Annotation(source_bytes, bytes(labels), tuple(captures))


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
    parser.add_argument("--spans", action="store_true", help="print byte ranges and labels")
    args = parser.parse_args(argv)
    annotation = annotate_python(args.path.read_bytes())
    if args.spans:
        for start, end, label, text in iter_spans(annotation):
            print(f"{start:6}:{end:<6} {label_name(label):20} {text!r}")
    else:
        print(colored(annotation), end="")


if __name__ == "__main__":
    main()
