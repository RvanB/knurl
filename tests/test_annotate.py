from neural_highlight.dataset.annotate import annotate_python
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
