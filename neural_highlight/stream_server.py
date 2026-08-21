"""Persistent JSON-lines server for low-latency ONNX highlighting."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import TextIO

from neural_highlight.labels import label_name
from neural_highlight.languages import LANGUAGE_NAMES
from neural_highlight.stream_infer import (
    OnnxStreamingSession,
    StreamPrediction,
    stream_highlight_onnx,
)


def prediction_spans(prediction: StreamPrediction) -> list[dict[str, int | str]]:
    """Return editor-ready spans with named syntax and language classes."""
    if not prediction.syntax:
        return []
    spans: list[dict[str, int | str]] = []
    start = 0
    for index in range(1, len(prediction.syntax) + 1):
        if (
            index == len(prediction.syntax)
            or prediction.syntax[index] != prediction.syntax[start]
            or prediction.regions[index] != prediction.regions[start]
        ):
            syntax_id = prediction.syntax[start]
            language_id = prediction.regions[start]
            spans.append({
                "start": start,
                "end": index,
                "syntax": label_name(syntax_id),
                "language": LANGUAGE_NAMES[language_id],
                "syntax_id": syntax_id,
                "language_id": language_id,
            })
            start = index
    return spans


def _source(request: dict[str, object]) -> bytes:
    if isinstance(request.get("text"), str):
        return request["text"].encode("utf-8")
    if isinstance(request.get("text_base64"), str):
        return base64.b64decode(request["text_base64"], validate=True)
    raise ValueError("request must contain string 'text' or 'text_base64'")


def serve(
    model: OnnxStreamingSession,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Process one request and emit one response for every non-empty input line."""
    for line in input_stream:
        if not line.strip():
            continue
        request_id: object = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            language = request.get("language", "unknown")
            if not isinstance(language, str):
                raise ValueError("'language' must be a string")
            source = _source(request)
            started = time.perf_counter()
            prediction = stream_highlight_onnx(model, source, language)
            response = {
                "id": request_id,
                "byte_length": len(source),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "spans": prediction_spans(prediction),
            }
        except Exception as error:
            response = {
                "id": request_id,
                "error": f"{type(error).__name__}: {error}",
            }
        output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
        output_stream.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--describe", action="store_true",
        help="write label/language ID mappings to stderr at startup",
    )
    args = parser.parse_args(argv)
    model = OnnxStreamingSession(args.model, args.device)
    if args.describe:
        print(
            json.dumps({
                "syntax": {
                    index: label_name(index)
                    for index in range(model.model_config.num_classes)
                },
                "languages": {index: name for index, name in enumerate(LANGUAGE_NAMES)},
                "chunk_length": model.chunk_length,
                "lookahead": model.lookahead,
            }, separators=(",", ":")),
            file=sys.stderr,
        )
    serve(model)


if __name__ == "__main__":
    main()
