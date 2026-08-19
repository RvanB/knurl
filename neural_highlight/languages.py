"""Stable IDs for host hints and per-byte language-region targets."""

LANGUAGE_NAMES = (
    "unknown",
    "prose",
    "python",
    "javascript",
    "typescript",
    "html",
    "css",
    "yaml",
    "jinja",
    "sql",
    "shell",
    "rust",
    "c",
    "c++",
    "go",
    "java",
    "markdown",
)
LANGUAGE_IDS = {name: index for index, name in enumerate(LANGUAGE_NAMES)}

