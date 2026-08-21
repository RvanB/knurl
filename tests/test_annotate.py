import pytest

from neural_highlight.dataset.annotate import (
    Annotation,
    annotate,
    annotate_python,
    colored,
    replace_comment_bodies,
    sanitize_comment_code,
)
from neural_highlight.labels import Label


def labels_for(annotation, token: str, occurrence: int = 0) -> set[Label]:
    encoded = token.encode()
    start = -1
    for _ in range(occurrence + 1):
        start = annotation.source.index(encoded, start + 1)
    return {Label(value) for value in annotation.labels[start : start + len(encoded)]}


def test_python_snippet_has_one_label_per_byte() -> None:
    source = "def café(x):\n    # note\n    return str(x.value)\n"
    annotation = annotate_python(source)
    assert len(annotation.labels) == len(source.encode("utf-8"))
    assert labels_for(annotation, "def") == {Label.KEYWORD}
    assert labels_for(annotation, "café") == {Label.FUNCTION}
    assert labels_for(annotation, "x") == {Label.PARAMETER}
    assert labels_for(annotation, "(") == {Label.PUNCTUATION}
    assert labels_for(annotation, "#") == {Label.COMMENT}
    comment_start = annotation.source.index(b"# note")
    assert annotation.label_mask[comment_start] == 1
    assert not any(annotation.label_mask[comment_start + 1 : comment_start + 6])
    assert labels_for(annotation, "return") == {Label.KEYWORD}
    assert labels_for(annotation, "value") == {Label.PROPERTY}


def test_empty_source() -> None:
    annotation = annotate_python(b"")
    assert annotation.labels == b""
    assert annotation.captures == ()


@pytest.mark.parametrize(
    ("language", "source", "token", "expected"),
    [
        ("javascript", "const x = 1;", "const", Label.KEYWORD),
        ("typescript", "const x: number = 1;", "number", Label.TYPE),
        ("html", '<div class="x">hi</div>', "div", Label.TYPE),
        ("css", ".x { color: red; }", "color", Label.PROPERTY),
        ("rust", "fn f() -> i32 { 1 }", "fn", Label.KEYWORD),
        ("c", "int f(void) { return 1; }", "return", Label.KEYWORD),
        ("c++", "class X { public: int f(); };", "class", Label.KEYWORD),
        ("go", "func f() int { return 1 }", "func", Label.KEYWORD),
        ("java", "class X { int f() { return 1; } }", "class", Label.KEYWORD),
        ("shell", "greet() { echo hello; }", "greet", Label.FUNCTION),
        ("sql", "SELECT COUNT(*) FROM users;", "COUNT", Label.FUNCTION),
        ("markdown", "# Heading\n", "#", Label.PUNCTUATION),
        ("lua", "local function greet(name) return name end", "name", Label.PARAMETER),
        ("ruby", "def greet(name); name; end", "greet", Label.FUNCTION),
        ("glsl", "uniform vec3 color;", "uniform", Label.KEYWORD),
    ],
)
def test_multilingual_teacher(language: str, source: str, token: str, expected: Label) -> None:
    assert labels_for(annotate(source, language), token) == {expected}


def test_block_comment_body_is_unsupervised_but_delimiters_are_kept() -> None:
    source = "/* fn hidden() { return 1; } prose */\nfn live() {}"
    annotation = annotate(source, "rust")
    opening = source.index("/*")
    closing = source.index("*/")
    hidden = source.index("fn hidden")
    live = source.index("fn live")
    assert annotation.label_mask[opening : opening + 2] == b"\x01\x01"
    assert annotation.label_mask[closing : closing + 2] == b"\x01\x01"
    assert not any(annotation.label_mask[hidden : hidden + len("fn hidden")])
    assert all(annotation.label_mask[live : live + len("fn live")])


def test_comment_replacement_removes_code_and_original_prose() -> None:
    source = """/*
This function returns the configured value.
```rust
fn hidden() { return 1; }
```
Callers should handle errors from this operation.
*/
fn live() {}
"""
    sanitized = sanitize_comment_code(source, "rust")
    assert b"fn hidden" not in sanitized
    assert b"configured value" not in sanitized
    assert b"handle errors" not in sanitized
    assert b"This " in sanitized or b"The " in sanitized
    assert b"fn live" in sanitized
    annotation = annotate(sanitized, "rust", supervise_comment_bodies=True)
    prose_start = sanitized.index(b"This ")
    assert annotation.labels[prose_start] == Label.COMMENT
    assert all(annotation.label_mask)


@pytest.mark.parametrize(
    ("language", "source", "opening", "closing"),
    [
        ("java", "/*\n * int hidden = call();\n */\nclass Live {}", b"/*", b"*/"),
        ("html", "<!--\n<script>hidden()</script>\n-->\n<div>live</div>", b"<!--", b"-->"),
        ("lua", "--[=[\nfunction hidden() end\n]=]\nlocal live = 1", b"--[=[", b"]=]"),
        ("ruby", "=begin\ndef hidden; end\n=end\nlive = 1", b"=begin", b"=end"),
    ],
)
def test_generated_comments_preserve_multiline_delimiters(
    language: str, source: str, opening: bytes, closing: bytes,
) -> None:
    replaced = replace_comment_bodies(source, language)
    assert opening in replaced and closing in replaced
    assert b"hidden" not in replaced
    assert b"live" in replaced.lower()
    assert replaced.count(b"\n") == source.encode().count(b"\n")


def test_generated_comments_preserve_shebang() -> None:
    source = b"#!/usr/bin/env bash\n# rm -rf something\necho live\n"
    replaced = replace_comment_bodies(source, "shell")
    assert replaced.startswith(b"#!/usr/bin/env bash\n")
    assert b"rm -rf" not in replaced
    assert b"echo live" in replaced


def test_identifier_and_comment_ansi_colors_are_distinct() -> None:
    annotation = Annotation(
        b"name note",
        bytes([Label.IDENTIFIER]) * 4 + bytes([Label.PLAIN])
        + bytes([Label.COMMENT]) * 4,
        (),
    )
    rendered = colored(annotation)
    assert "\033[38;5;223mname\033[0m" in rendered
    assert "\033[90mnote\033[0m" in rendered


def test_function_ansi_colors_use_bright_sky_blue() -> None:
    annotation = Annotation(
        b"call define",
        bytes([Label.FUNCTION]) * 4 + bytes([Label.PLAIN])
        + bytes([Label.FUNCTION]) * 6,
        (),
    )
    rendered = colored(annotation)
    assert "\033[38;5;117mcall\033[0m" in rendered
    assert "\033[38;5;117mdefine\033[0m" in rendered
