"""Stateful inference over arbitrarily long byte streams."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from neural_highlight.dataset.annotate import Annotation, colored
from neural_highlight.dataset.fragments import LANGUAGE_IDS, PAD_BYTE_ID
from neural_highlight.labels import label_name
from neural_highlight.languages import LANGUAGE_NAMES
from neural_highlight.models.streaming_gru import StreamingByteGRU, StreamingGRUConfig


@dataclass(frozen=True)
class StreamPrediction:
    syntax: bytes
    regions: bytes
    checkpoints: tuple[torch.Tensor, ...]


def load_streaming_model(path: Path, device: torch.device) -> StreamingByteGRU:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = StreamingByteGRU(StreamingGRUConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


@torch.inference_mode()
def stream_highlight(
    model: StreamingByteGRU,
    source: bytes,
    language: str = "unknown",
    chunk_length: int = 256,
    lookahead: int = 128,
) -> StreamPrediction:
    if not source:
        return StreamPrediction(b"", b"", ())
    device = next(model.parameters()).device
    state = model.initial_state(1, device)
    language_id = torch.tensor([LANGUAGE_IDS.get(language, 0)], device=device)
    syntax_parts: list[bytes] = []
    region_parts: list[bytes] = []
    checkpoints: list[torch.Tensor] = [state.squeeze(1).cpu()]
    for offset in range(0, len(source), chunk_length):
        committed = min(chunk_length, len(source) - offset)
        raw = source[offset : offset + committed + lookahead]
        values = list(raw) + [PAD_BYTE_ID] * (chunk_length + lookahead - len(raw))
        input_ids = torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)
        syntax_logits, region_logits, state = model.forward_chunk(
            input_ids, state, committed, language_id
        )
        syntax_parts.append(bytes(syntax_logits.argmax(-1).squeeze(0).cpu().tolist()))
        region_parts.append(bytes(region_logits.argmax(-1).squeeze(0).cpu().tolist()))
        checkpoints.append(state.squeeze(1).cpu())
    return StreamPrediction(
        b"".join(syntax_parts), b"".join(region_parts), tuple(checkpoints)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source", nargs="?", help="source file or '-' for stdin")
    parser.add_argument("--text")
    parser.add_argument("--language", default="unknown")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-length", type=int)
    parser.add_argument("--lookahead", type=int)
    parser.add_argument("--color", action="store_true")
    args = parser.parse_args(argv)
    if (args.source is None) == (args.text is None):
        parser.error("provide exactly one source file/stdin marker or --text")
    if args.text is not None:
        source = args.text.encode("utf-8")
    elif args.source == "-":
        source = sys.stdin.buffer.read()
    else:
        source = Path(args.source).read_bytes()
    device = torch.device(args.device)
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train_config = raw_checkpoint.get("train_config", {})
    chunk_length = args.chunk_length or train_config.get("chunk_length", 256)
    lookahead = args.lookahead if args.lookahead is not None else train_config.get("lookahead", 128)
    prediction = stream_highlight(
        load_streaming_model(args.checkpoint, device), source, args.language,
        chunk_length, lookahead,
    )
    if args.color:
        print(colored(Annotation(source, prediction.syntax, ())), end="")
        return
    start = 0
    for index in range(1, len(source) + 1):
        if (
            index == len(source)
            or prediction.syntax[index] != prediction.syntax[start]
            or prediction.regions[index] != prediction.regions[start]
        ):
            print(
                f"{start:6}:{index:<6} {LANGUAGE_NAMES[prediction.regions[start]]:12} "
                f"{label_name(prediction.syntax[start]):20} {source[start:index]!r}"
            )
            start = index


if __name__ == "__main__":
    main()
