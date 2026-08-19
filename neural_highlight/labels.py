"""The small, cross-language label vocabulary used by models and datasets."""

from __future__ import annotations

from enum import IntEnum


class Label(IntEnum):
    PLAIN = 0
    KEYWORD = 1
    IDENTIFIER = 2
    FUNCTION = 3
    FUNCTION_DEFINITION = 4
    PARAMETER = 5
    TYPE = 6
    PROPERTY = 7
    STRING = 8
    COMMENT = 9
    NUMBER = 10
    OPERATOR = 11
    PUNCTUATION = 12
    CONSTANT = 13


LABEL_NAMES = tuple(label.name.lower() for label in Label)


def label_name(value: int | Label) -> str:
    """Return the serialized lowercase name for a label value."""
    return Label(value).name.lower()

