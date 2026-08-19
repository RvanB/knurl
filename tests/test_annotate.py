import pytest

from neural_highlight.dataset.annotate import annotate, annotate_python
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
    assert labels_for(annotation, "café") == {Label.FUNCTION_DEFINITION}
    assert labels_for(annotation, "x") == {Label.PARAMETER}
    assert labels_for(annotation, "(") == {Label.PUNCTUATION}
    assert labels_for(annotation, "# note") == {Label.COMMENT}
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
    ],
)
def test_multilingual_teacher(language: str, source: str, token: str, expected: Label) -> None:
    assert labels_for(annotate(source, language), token) == {expected}
