"""Run a trained BiGRU directly on source bytes (without Tree-sitter)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_highlight.dataset.fragments import LANGUAGE_IDS
from neural_highlight.labels import Label, label_name
from neural_highlight.languages import LANGUAGE_NAMES
from neural_highlight.models.bigru import BiGRUConfig, ByteBiGRU


def load_model(path: Path, device: torch.device) -> ByteBiGRU:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ByteBiGRU(BiGRUConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


@torch.inference_mode()
def highlight(model: ByteBiGRU, source: bytes, language: str = "unknown") -> bytes:
    return analyze(model, source, language)[0]


@torch.inference_mode()
def analyze(model: ByteBiGRU, source: bytes, language: str = "unknown") -> tuple[bytes, bytes]:
    """Return syntax and local-language predictions for every source byte."""
    if not source:
        return b"", b""
    device = next(model.parameters()).device
    input_ids = torch.tensor(list(source), dtype=torch.long, device=device).unsqueeze(0)
    language_id = torch.tensor([LANGUAGE_IDS.get(language, 0)], device=device)
    syntax_logits, region_logits = model.forward_with_regions(input_ids, language_id)
    syntax = bytes(syntax_logits.argmax(dim=-1).squeeze(0).cpu().tolist())
    regions = bytes(region_logits.argmax(dim=-1).squeeze(0).cpu().tolist())
    return syntax, regions


def spans(source: bytes, labels: bytes):
    if not source:
        return
    start = 0
    for index in range(1, len(source) + 1):
        if index == len(source) or labels[index] != labels[start]:
            yield start, index, Label(labels[start]), source[start:index]
            start = index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--language", default="python")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    source = args.source.read_bytes()
    labels, regions = analyze(
        load_model(args.checkpoint, torch.device(args.device)), source, args.language
    )
    start = 0
    for index in range(1, len(source) + 1):
        if index == len(source) or labels[index] != labels[start] or regions[index] != regions[start]:
            print(
                f"{start:6}:{index:<6} {LANGUAGE_NAMES[regions[start]]:12} "
                f"{label_name(labels[start]):20} {source[start:index]!r}"
            )
            start = index


if __name__ == "__main__":
    main()
