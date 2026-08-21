"""Stateful inference over arbitrarily long byte streams."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from neural_highlight.dataset.annotate import Annotation, colored
from neural_highlight.dataset.fragments import LANGUAGE_IDS, PAD_BYTE_ID
from neural_highlight.labels import LABEL_SCHEMA_VERSION, label_name
from neural_highlight.languages import LANGUAGE_NAMES, LANGUAGE_SCHEMA_VERSION
from neural_highlight.models.streaming_gru import StreamingByteGRU, StreamingGRUConfig

ort.disable_telemetry_events()


@dataclass(frozen=True)
class StreamPrediction:
    syntax: bytes
    regions: bytes
    checkpoints: tuple[torch.Tensor, ...]


class AlphabeticRunDecoder:
    """Assign one pooled syntax class to each contiguous ASCII letter run."""

    def __init__(self) -> None:
        self.labels = bytearray()
        self._run_start: int | None = None
        self._run_scores: torch.Tensor | None = None

    def _finish_run(self) -> None:
        if self._run_start is None or self._run_scores is None:
            return
        run_length = len(self.labels) - self._run_start
        label = int(self._run_scores.argmax())
        self.labels[self._run_start :] = bytes([label]) * (
            run_length
        )
        self._run_start = None
        self._run_scores = None

    def extend(self, source: bytes, logits: torch.Tensor) -> None:
        if logits.shape[0] != len(source):
            raise ValueError("source and syntax logits must have equal lengths")
        logits = logits.detach().cpu()
        predictions = logits.argmax(dim=-1).tolist()
        for value, prediction, scores in zip(source, predictions, logits):
            if (ord("A") <= value <= ord("Z")) or (ord("a") <= value <= ord("z")):
                self.labels.append(prediction)
                if self._run_start is None:
                    self._run_start = len(self.labels) - 1
                    self._run_scores = scores.clone()
                else:
                    self._run_scores += scores
            else:
                self._finish_run()
                self.labels.append(prediction)

    def finish(self) -> bytes:
        self._finish_run()
        return bytes(self.labels)


def stabilize_alphabetic_runs(source: bytes, logits: torch.Tensor) -> bytes:
    """Convenience wrapper for applying the streaming word-consistency decoder."""
    decoder = AlphabeticRunDecoder()
    decoder.extend(source, logits)
    return decoder.finish()


def load_streaming_model(path: Path, device: torch.device) -> StreamingByteGRU:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    checkpoint_schema = checkpoint.get("label_schema_version", 1)
    if checkpoint_schema != LABEL_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint uses label schema {checkpoint_schema}, expected "
            f"{LABEL_SCHEMA_VERSION}; retrain with the current label vocabulary"
        )
    language_schema = checkpoint.get("language_schema_version", 1)
    if language_schema != LANGUAGE_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint uses language schema {language_schema}, expected "
            f"{LANGUAGE_SCHEMA_VERSION}; retrain with the current language vocabulary"
        )
    model = StreamingByteGRU(StreamingGRUConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


class OnnxStreamingSession:
    """Validated ONNX Runtime session for the fixed streaming chunk contract."""

    def __init__(self, path: Path, device: str = "cpu") -> None:
        available = set(ort.get_available_providers())
        if device.startswith("cuda"):
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "ONNX Runtime CUDAExecutionProvider is unavailable; use --device cpu "
                    "or install a CUDA-compatible onnxruntime-gpu build"
                )
            device_id = int(device.partition(":")[2] or 0)
            providers: list[object] = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError("ONNX inference device must be 'cpu' or 'cuda[:index]'")
        self.session = ort.InferenceSession(str(path), providers=providers)
        metadata = self.session.get_modelmeta().custom_metadata_map
        if int(metadata.get("label_schema_version", 1)) != LABEL_SCHEMA_VERSION:
            raise ValueError("ONNX model uses an incompatible label schema")
        if int(metadata.get("language_schema_version", 1)) != LANGUAGE_SCHEMA_VERSION:
            raise ValueError("ONNX model uses an incompatible language schema")
        if metadata.get("architecture") != "streaming-gru":
            raise ValueError("ONNX model is not a streaming GRU export")
        self.chunk_length = int(metadata["chunk_length"])
        self.lookahead = int(metadata["lookahead"])
        self.model_config = StreamingGRUConfig(**json.loads(metadata["model_config"]))

    def initial_state(self) -> np.ndarray:
        return np.zeros(
            (self.model_config.num_layers, 1, self.model_config.hidden_size),
            dtype=np.float32,
        )

    def forward_chunk(
        self, input_ids: np.ndarray, state: np.ndarray, language_id: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        syntax, regions, state_out = self.session.run(
            None,
            {"input_ids": input_ids, "state_in": state, "language_id": language_id},
        )
        return syntax, regions, state_out


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
    syntax_decoder = AlphabeticRunDecoder()
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
        syntax_decoder.extend(raw[:committed], syntax_logits.squeeze(0))
        region_parts.append(bytes(region_logits.argmax(-1).squeeze(0).cpu().tolist()))
        checkpoints.append(state.squeeze(1).cpu())
    return StreamPrediction(
        syntax_decoder.finish(), b"".join(region_parts), tuple(checkpoints)
    )


def stream_highlight_onnx(
    model: OnnxStreamingSession,
    source: bytes,
    language: str = "unknown",
) -> StreamPrediction:
    if not source:
        return StreamPrediction(b"", b"", ())
    state = model.initial_state()
    language_id = np.asarray([LANGUAGE_IDS.get(language, 0)], dtype=np.int64)
    syntax_decoder = AlphabeticRunDecoder()
    region_parts: list[bytes] = []
    checkpoints: list[torch.Tensor] = [torch.from_numpy(state[:, 0].copy())]
    input_length = model.chunk_length + model.lookahead
    for offset in range(0, len(source), model.chunk_length):
        committed = min(model.chunk_length, len(source) - offset)
        raw = source[offset : offset + committed + model.lookahead]
        inputs = np.full((1, input_length), PAD_BYTE_ID, dtype=np.int64)
        inputs[0, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        syntax_logits, region_logits, state = model.forward_chunk(
            inputs, state, language_id,
        )
        syntax_decoder.extend(
            raw[:committed], torch.from_numpy(syntax_logits[0, :committed]),
        )
        region_parts.append(bytes(region_logits[0, :committed].argmax(-1).tolist()))
        checkpoints.append(torch.from_numpy(state[:, 0].copy()))
    return StreamPrediction(
        syntax_decoder.finish(), b"".join(region_parts), tuple(checkpoints),
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
    if args.checkpoint.suffix.lower() != ".onnx":
        raise ValueError(
            "streaming inference now requires an ONNX model; convert a checkpoint with "
            "scripts/export_streaming_onnx.py"
        )
    model = OnnxStreamingSession(args.checkpoint, args.device)
    if args.chunk_length is not None and args.chunk_length != model.chunk_length:
        raise ValueError("--chunk-length must match the fixed ONNX export shape")
    if args.lookahead is not None and args.lookahead != model.lookahead:
        raise ValueError("--lookahead must match the fixed ONNX export shape")
    prediction = stream_highlight_onnx(model, source, args.language)
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
