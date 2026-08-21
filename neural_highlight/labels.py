"""The small, cross-language label vocabulary used by models and datasets."""

from __future__ import annotations

from enum import IntEnum


class Label(IntEnum):
    PLAIN = 0
    KEYWORD = 1
    IDENTIFIER = 2
    FUNCTION = 3
    PARAMETER = 4
    TYPE = 5
    PROPERTY = 6
    STRING = 7
    COMMENT = 8
    NUMBER = 9
    OPERATOR = 10
    PUNCTUATION = 11
    CONSTANT = 12


class Enclosure(IntEnum):
    CODE = 0
    STRING = 1
    COMMENT = 2


ENCLOSURE_NAMES = tuple(value.name.lower() for value in Enclosure)


LABEL_NAMES = tuple(label.name.lower() for label in Label)
LABEL_SCHEMA_VERSION = 3


def label_name(value: int | Label) -> str:
    """Return the serialized lowercase name for a label value."""
    return Label(value).name.lower()
