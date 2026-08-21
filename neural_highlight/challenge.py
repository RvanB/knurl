"""Deterministic behavioral validation for streaming highlighting."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_highlight.dataset.annotate import annotate
from neural_highlight.dataset.comment_wrapping import wrap_code_as_comment
from neural_highlight.dataset.string_wrapping import wrap_code_as_string
from neural_highlight.dataset.fragments import PAD_BYTE_ID
from neural_highlight.labels import Label
from neural_highlight.languages import LANGUAGE_IDS, LANGUAGE_NAMES
from neural_highlight.metrics import ClassificationMetrics
from neural_highlight.models.streaming_gru import StreamingByteGRU


@dataclass(frozen=True)
class ChallengeDocument:
    source: bytes
    labels: bytes
    regions: bytes
    slices: dict[str, bytes]


def _code(source: bytes, language: str) -> tuple[bytes, bytes, bytes, bytes]:
    annotation = annotate(source, language, supervise_comment_bodies=True)
    region = LANGUAGE_IDS[language]
    return source, annotation.labels, bytes([region]) * len(source), bytes([1]) * len(source)


def _wrapped(source: bytes, language: str, style: str) -> tuple[bytes, bytes, bytes]:
    raw, labels, regions, mask = _code(source, language)
    wrapped = wrap_code_as_comment(raw, labels, regions, mask, 4096, style=style)
    return wrapped.source, wrapped.labels, wrapped.regions


def challenge_document() -> ChallengeDocument:
    """Build fixed prose, embedded-code, ordinary-code, and transition cases."""
    pieces: list[tuple[bytes, bytes, bytes, str]] = []

    prose_specs = (
        (b"/*\n * This explanation describes the configured value in plain language.\n */\n", "java"),
        (b"# Retry the request after the cache has been refreshed.\n", "python"),
        (b"// Keep this branch simple so failures remain easy to diagnose.\n", "javascript"),
        (b"-- Convert the stored timestamp before displaying it to the user.\n", "lua"),
        (b"<!-- The navigation is intentionally hidden on compact screens. -->\n", "html"),
        (b"/* The Java class returns the configured value after initialization. */\n", "java"),
        (b"// Call the initialization function after loading the uniform values.\n", "javascript"),
        (b"# This function returns a type name rather than executing code.\n", "python"),
        (b"/* Now some C code appears in the following section of this comment. */\n", "c"),
    )
    for prose, language in prose_specs:
        pieces.append((
            prose, bytes([Label.COMMENT]) * (len(prose) - 1) + bytes([Label.PLAIN]),
            bytes([LANGUAGE_IDS[language]]) * len(prose), "prose",
        ))
    wrappers = (
        (b"public static void main(String[] args) {\n    System.out.println(\"hello\");\n}", "java", "c_starred"),
        (b"int main(int argc, char **argv) {\n    printf(\"hello\\n\");\n    return 0;\n}", "c", "c_multiline"),
        (b"<section class=\"note\">value</section>", "html", "html_multiline"),
        (b"def calculate(value):\n    return value + 1", "python", "hash_lines"),
        (b"function calculate(value) {\n    return value + 1;\n}", "javascript", "slash_lines"),
        (b"SELECT account_id, balance\nFROM accounts\nWHERE active = true;", "sql", "c_starred"),
        (b"fn clamp(value: f32) -> f32 { value.max(0.0).min(1.0) }", "rust", "slash_lines"),
        (b"local function greet(name)\n  return 'hello ' .. name\nend", "lua", "hash_lines"),
    )
    for source, language, style in wrappers:
        wrapped_source, wrapped_labels, wrapped_regions = _wrapped(source, language, style)
        pieces.append((wrapped_source, wrapped_labels, wrapped_regions, "wrapped"))
        # The following zero-length-source pieces carry additional masks below.
        pieces.append((b"\n", bytes([Label.PLAIN]), bytes([LANGUAGE_IDS[language]]), "separator"))

    embedded_strings = (
        (b"vec4 shade(vec2 uv) {\n    uniform float exposure;\n    return vec4(uv, exposure, 1.0);\n}",
         "glsl", "python_double", "python"),
        (b"SELECT customer_id, COUNT(*) AS orders\nFROM purchases\nGROUP BY customer_id;",
         "sql", "backtick", "javascript"),
        (b"<article class=\"card\"><h2>Title</h2><p>Body</p></article>",
         "html", "backtick", "javascript"),
        (b"body { color: #eee; background: rgb(20, 24, 30); }",
         "css", "python_single", "python"),
    )
    for source, language, style, host in embedded_strings:
        raw, labels, regions, mask = _code(source, language)
        wrapped = wrap_code_as_string(
            raw, labels, regions, mask, 4096, style=style,
            host_region=LANGUAGE_IDS[host],
        )
        pieces.append((wrapped.source, wrapped.labels, wrapped.regions, "string_wrapped"))
        pieces.append((b"\n", bytes([Label.PLAIN]), bytes([LANGUAGE_IDS[host]]), "separator"))

    ordinary_specs = (
        (b"def greet(name):\n    print(name)\n", "python"),
        (b"public int size() { return 3; }\n", "java"),
        (b"fn size() -> usize { 3 }\n", "rust"),
        (b"const size = (items) => items.length;\n", "javascript"),
        (b"func size[T any](items []T) int { return len(items) }\n", "go"),
        (b"SELECT name FROM users WHERE enabled = TRUE;\n", "sql"),
    )
    for source, language in ordinary_specs:
        raw, labels, regions, _ = _code(source, language)
        pieces.append((raw, labels, regions, "ordinary_code"))

    long_strings = (
        b'description = """\n' + b"The quick brown fox jumps over the lazy dog.\n" * 12 + b'"""\n',
        b'wrapped = """\n' + b"The quick brown fox jumps over\nthe lazy dog and then walks back\n" * 10 + b'"""\n',
        b'notes = """\n' + b"function returns value when cache entry exists\notherwise request fresh data and retry later\n" * 8 + b'"""\n',
        b'message = """\n' + b"Names dates numbers and ordinary words can continue\nacross many lines without changing their meaning\n" * 8 + b'"""\n',
    )
    for long_string in long_strings:
        raw, labels, regions, _ = _code(long_string, "python")
        pieces.append((raw, labels, regions, "long_string"))

    source = b"".join(piece[0] for piece in pieces)
    labels = bytes(
        Label.PLAIN if label in (Label.STRING, Label.COMMENT) else label
        for label in b"".join(piece[1] for piece in pieces)
    )
    regions = b"".join(piece[2] for piece in pieces)
    slices = {name: bytearray(len(source)) for name in (
        "prose", "embedded_code", "embedded_string_code", "decoration", "ordinary_code",
        "long_string", "language_transition",
    )}
    offset = 0
    for piece_source, piece_labels, _, kind in pieces:
        end = offset + len(piece_source)
        if kind == "wrapped":
            for index, label in enumerate(piece_labels, offset):
                slices["decoration" if label == Label.COMMENT else "embedded_code"][index] = 1
        elif kind == "string_wrapped":
            for index, label in enumerate(piece_labels, offset):
                if label == Label.STRING:
                    slices["decoration"][index] = 1
                else:
                    slices["embedded_string_code"][index] = 1
        elif kind == "long_string":
            for index, label in enumerate(piece_labels, offset):
                if label == Label.STRING:
                    slices["long_string"][index] = 1
        elif kind in slices:
            slices[kind][offset:end] = bytes([1]) * len(piece_source)
        offset = end
    for index in range(1, len(regions)):
        if regions[index] != regions[index - 1]:
            start, end = max(0, index - 16), min(len(source), index + 16)
            slices["language_transition"][start:end] = bytes([1]) * (end - start)
    return ChallengeDocument(source, labels, regions, {k: bytes(v) for k, v in slices.items()})


def evaluate_challenge(
    model: StreamingByteGRU,
    device: torch.device,
    chunk_length: int,
    lookahead: int,
) -> dict[str, object]:
    document = challenge_document()
    syntax_predictions: list[torch.Tensor] = []
    region_predictions: list[torch.Tensor] = []
    state = model.initial_state(1, device)
    unknown = torch.tensor([LANGUAGE_IDS["unknown"]], device=device)
    for start in range(0, len(document.source), chunk_length):
        committed = min(chunk_length, len(document.source) - start)
        raw = document.source[start : start + committed + lookahead]
        inputs = torch.full((1, chunk_length + lookahead), PAD_BYTE_ID, dtype=torch.long, device=device)
        inputs[0, : len(raw)] = torch.tensor(list(raw), device=device)
        syntax, regions, state = model.forward_chunk(inputs, state, chunk_length, unknown)
        syntax_predictions.append(syntax[0, :committed].argmax(-1).cpu())
        region_predictions.append(regions[0, :committed].argmax(-1).cpu())
    syntax_prediction = torch.cat(syntax_predictions)
    region_prediction = torch.cat(region_predictions)
    syntax_target = torch.tensor(list(document.labels))
    region_target = torch.tensor(list(document.regions))
    results: dict[str, object] = {}
    for name, raw_mask in document.slices.items():
        mask = torch.tensor(list(raw_mask), dtype=torch.bool)
        metric = ClassificationMetrics(
            model.config.num_languages if name == "language_transition" else model.config.num_classes,
            LANGUAGE_NAMES if name == "language_transition" else None,
        )
        metric.update_predictions(
            region_prediction[mask] if name == "language_transition" else syntax_prediction[mask],
            region_target[mask] if name == "language_transition" else syntax_target[mask],
        )
        computed = metric.compute()
        results[name] = computed
    weights = {
        "embedded_code": 0.40,
        "embedded_string_code": 0.20,
        "language_transition": 0.15,
        "long_string": 0.10,
        "ordinary_code": 0.05,
        "prose": 0.05,
        "decoration": 0.05,
    }
    # State-heavy slices are mostly one target class, where byte accuracy is
    # more stable and meaningful than the macro-F1 of whichever classes happen
    # to occur. Code-bearing slices retain macro-F1.
    score_components = {
        name: float(results[name][
            "accuracy" if name in {"long_string", "prose", "decoration"} else "macro_f1"
        ])
        for name in weights
    }
    results["score_components"] = score_components
    results["score"] = sum(weights[name] * score_components[name] for name in weights)
    return results
