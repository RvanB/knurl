"""Annotate complete Python files with one normalized label per UTF-8 byte."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable
import hashlib
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
import tree_sitter_bash
import tree_sitter_glsl
import tree_sitter_lua
import tree_sitter_markdown
import tree_sitter_ruby
import tree_sitter_sql
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
_MARKDOWN_QUERY = (
    files("tree_sitter_markdown")
    .joinpath("queries/markdown/highlights.scm")
    .read_text(encoding="utf-8")
)
_GLSL_QUERY = (
    files("tree_sitter_glsl").joinpath("queries/highlights.scm").read_text(encoding="utf-8")
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
    "shell": (tree_sitter_bash.language, tree_sitter_bash.HIGHLIGHTS_QUERY),
    "sql": (tree_sitter_sql.language, tree_sitter_sql.HIGHLIGHTS_QUERY),
    "markdown": (tree_sitter_markdown.language, _MARKDOWN_QUERY),
    "lua": (tree_sitter_lua.language, tree_sitter_lua.HIGHLIGHTS_QUERY),
    "ruby": (tree_sitter_ruby.language, tree_sitter_ruby.HIGHLIGHTS_QUERY),
    "glsl": (tree_sitter_glsl.language, _GLSL_QUERY),
}
SUPPORTED_LANGUAGES = tuple(_LANGUAGE_SPECS)
_COMPILED: dict[str, tuple[Language, Query]] = {}
_COMMENT_SUBJECTS = (
    b"This section", b"This note", b"The documentation", b"The description",
    b"This explanation", b"The following guidance", b"This paragraph", b"The context",
)
_COMMENT_VERBS = (
    b"describes", b"documents", b"clarifies", b"explains", b"records", b"summarizes",
)
_COMMENT_OBJECTS = (
    b"the expected behavior", b"the relevant configuration", b"the intended usage",
    b"the processing requirements", b"the important details", b"the normal operation",
    b"the surrounding context", b"the maintenance considerations",
)
_COMMENT_AUDIENCES = (
    b"for callers", b"for maintainers", b"for future changes", b"during normal use",
    b"for reviewers", b"when updating this component",
)


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
    lua = re.match(rb"--\[(=*)\[", source)
    if lua is not None:
        closing = b"]" + lua.group(1) + b"]"
        return lua.end(), len(closing) if source.endswith(closing) else 0
    for opening, closing in (
        (b"<!--", b"-->"),
        (b"--[[", b"]]"),
        (b"=begin", b"=end"),
        (b"/**", b"*/"),
        (b"/*!", b"*/"),
        (b"/*", b"*/"),
        (b"///", b""),
        (b"//!", b""),
        (b"//", b""),
        (b"--", b""),
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


def _generated_comment_line(target_length: int, variant: int) -> bytes:
    if target_length < 12:
        return (b"Note.", b"Details.", b"Context.")[variant % 3]
    subject = _COMMENT_SUBJECTS[variant % len(_COMMENT_SUBJECTS)]
    verb = _COMMENT_VERBS[(variant // 3) % len(_COMMENT_VERBS)]
    object_ = _COMMENT_OBJECTS[(variant // 7) % len(_COMMENT_OBJECTS)]
    audience = _COMMENT_AUDIENCES[(variant // 11) % len(_COMMENT_AUDIENCES)]
    sentence = b" ".join((subject, verb, object_, audience)) + b"."
    if target_length < 36:
        return b" ".join((subject, verb, b"the details."))
    return sentence


def replace_comment_bodies(source: str | bytes, language: str) -> bytes:
    """Replace captured comment bodies with deterministic, code-free prose."""
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
        if capture.start_byte == 0 and captured.startswith(b"#!"):
            output.extend(captured)
            cursor = capture.end_byte
            continue
        opening_length, closing_length = _comment_delimiters(captured)
        if not opening_length:
            # Preserve an unknown delimiter rather than risk producing invalid source.
            output.extend(captured)
            cursor = capture.end_byte
            continue
        opening = captured[:opening_length]
        closing = captured[len(captured) - closing_length :] if closing_length else b""
        body_end = len(captured) - closing_length if closing_length else len(captured)
        body = captured[opening_length:body_end]
        digest = hashlib.sha256(
            source_bytes[max(0, capture.start_byte - 32) : capture.end_byte + 32]
        ).digest()
        generated: list[bytes] = []
        for line_index, line in enumerate(body.splitlines(keepends=True)):
            content = line.rstrip(b"\r\n")
            ending = line[len(content) :]
            prefix_match = re.match(rb"\s*(?:\*+\s*)?", content)
            prefix = prefix_match.group(0) if prefix_match is not None else b""
            original_text = content[len(prefix) :].strip()
            if not original_text:
                generated.append(prefix + ending)
                continue
            sentence = _generated_comment_line(len(original_text), digest[line_index % len(digest)])
            separator = b"" if prefix.endswith((b" ", b"\t")) or not prefix else b" "
            generated.append(prefix + separator + sentence + ending)
        if body and not generated:
            generated.append(_generated_comment_line(len(body), digest[0]))
        output.extend(opening + b"".join(generated) + closing)
        cursor = capture.end_byte
    output.extend(source_bytes[cursor:])
    return bytes(output)


def sanitize_comment_code(source: str | bytes, language: str) -> bytes:
    """Compatibility alias for the generated-comment preparation policy."""
    return replace_comment_bodies(source, language)


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
    # Pale tan stays legible on dark terminals and is visibly distinct from
    # the subdued gray used for comments.
    Label.IDENTIFIER: "\033[38;5;223m",
    Label.FUNCTION: "\033[38;5;117m",
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
