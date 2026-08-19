from neural_highlight.dataset.normalize import normalize_capture
from neural_highlight.labels import Label


def test_capture_normalization_is_deterministic() -> None:
    assert normalize_capture("@function.call") is Label.FUNCTION
    assert normalize_capture("variable.parameter") is Label.PARAMETER
    assert normalize_capture("type.builtin") is Label.TYPE
    assert normalize_capture("unrecognized.capture") is Label.PLAIN

