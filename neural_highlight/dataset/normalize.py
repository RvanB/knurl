"""Normalize Tree-sitter highlight capture names into the project ontology."""

from __future__ import annotations

from neural_highlight.labels import Label


# Ordered from specific to general. Tree-sitter captures are hierarchical names,
# but grammar authors do not use an entirely uniform naming scheme.
_EXACT = {
    "function.definition": Label.FUNCTION_DEFINITION,
    "function.method.definition": Label.FUNCTION_DEFINITION,
    "variable.parameter": Label.PARAMETER,
    "type.builtin": Label.TYPE,
    "variable.member": Label.PROPERTY,
    "property": Label.PROPERTY,
    "attribute": Label.PROPERTY,
    "tag": Label.TYPE,
    "constructor": Label.TYPE,
    "type.definition": Label.TYPE,
    "constant.builtin": Label.CONSTANT,
    "boolean": Label.CONSTANT,
}

_PREFIXES = (
    ("comment", Label.COMMENT),
    ("string", Label.STRING),
    ("number", Label.NUMBER),
    ("keyword", Label.KEYWORD),
    ("operator", Label.OPERATOR),
    ("punctuation", Label.PUNCTUATION),
    ("delimiter", Label.PUNCTUATION),
    ("bracket", Label.PUNCTUATION),
    ("function", Label.FUNCTION),
    ("method", Label.FUNCTION),
    ("type", Label.TYPE),
    ("constant", Label.CONSTANT),
    ("variable", Label.IDENTIFIER),
    ("identifier", Label.IDENTIFIER),
    ("label", Label.IDENTIFIER),
    ("namespace", Label.IDENTIFIER),
    ("module", Label.IDENTIFIER),
    ("preproc", Label.KEYWORD),
)


def normalize_capture(capture: str) -> Label:
    """Map a capture such as ``function.call`` to a universal label."""
    capture = capture.removeprefix("@").lower()
    if capture in _EXACT:
        return _EXACT[capture]
    for prefix, label in _PREFIXES:
        if capture == prefix or capture.startswith(prefix + "."):
            return label
    return Label.PLAIN
