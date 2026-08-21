"""Export and verify a stateful streaming GRU checkpoint as ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from neural_highlight.labels import LABEL_SCHEMA_VERSION
from neural_highlight.languages import LANGUAGE_SCHEMA_VERSION
from neural_highlight.models.streaming_gru import StreamingByteGRU, StreamingGRUConfig

ort.disable_telemetry_events()


class _OnnxStreamingWrapper(nn.Module):
    def __init__(self, model: StreamingByteGRU, commit_length: int) -> None:
        super().__init__()
        self.model = model
        self.commit_length = commit_length

    def forward(
        self, input_ids: Tensor, state_in: Tensor, language_id: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.model.forward_chunk(
            input_ids, state_in, self.commit_length, language_id,
        )


def _validate_checkpoint(checkpoint: dict[str, object]) -> None:
    label_schema = checkpoint.get("label_schema_version", 1)
    if label_schema != LABEL_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint uses label schema {label_schema}, expected {LABEL_SCHEMA_VERSION}"
        )
    language_schema = checkpoint.get("language_schema_version", 1)
    if language_schema != LANGUAGE_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint uses language schema {language_schema}, expected "
            f"{LANGUAGE_SCHEMA_VERSION}"
        )


def _verify_export(
    model: StreamingByteGRU, output: Path, chunk_length: int, lookahead: int,
) -> None:
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(20260819)
    torch_state = model.initial_state(1)
    onnx_state = torch_state.numpy()
    language = torch.tensor([2], dtype=torch.long)
    for _ in range(2):
        inputs = torch.randint(
            0, 256, (1, chunk_length + lookahead), generator=generator,
        )
        with torch.inference_mode():
            expected_syntax, expected_regions, torch_state = model.forward_chunk(
                inputs, torch_state, chunk_length, language,
            )
        actual_syntax, actual_regions, onnx_state = session.run(
            None,
            {
                "input_ids": inputs.numpy(),
                "state_in": onnx_state,
                "language_id": language.numpy(),
            },
        )
        np.testing.assert_allclose(actual_syntax, expected_syntax.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(actual_regions, expected_regions.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(onnx_state, torch_state.numpy(), rtol=1e-4, atol=1e-5)


def export_streaming_checkpoint(
    checkpoint_path: Path,
    output_path: Path | None = None,
    *,
    verify: bool = True,
) -> Path:
    """Export ``checkpoint_path`` and optionally verify two recurrent chunks."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint(checkpoint)
    model_config = StreamingGRUConfig(**checkpoint["model_config"])
    model = StreamingByteGRU(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    train_config = checkpoint.get("train_config", {})
    chunk_length = int(train_config.get("chunk_length", 256))
    lookahead = int(train_config.get("lookahead", 128))
    wrapper = _OnnxStreamingWrapper(model, chunk_length).eval()
    inputs = torch.zeros((1, chunk_length + lookahead), dtype=torch.long)
    state = model.initial_state(1)
    language = torch.zeros((1,), dtype=torch.long)
    output_path = output_path or checkpoint_path.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (inputs, state, language),
        output_path,
        input_names=("input_ids", "state_in", "language_id"),
        output_names=("syntax_logits", "region_logits", "state_out"),
        opset_version=17,
        dynamo=False,
    )
    exported = onnx.load(output_path)
    metadata = {
        "architecture": "streaming-gru",
        "label_schema_version": str(LABEL_SCHEMA_VERSION),
        "language_schema_version": str(LANGUAGE_SCHEMA_VERSION),
        "chunk_length": str(chunk_length),
        "lookahead": str(lookahead),
        "model_config": json.dumps(model_config.to_dict(), separators=(",", ":")),
    }
    del exported.metadata_props[:]
    for key, value in metadata.items():
        entry = exported.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(exported)
    onnx.save(exported, output_path)
    if verify:
        _verify_export(model, output_path, chunk_length, lookahead)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)
    print(export_streaming_checkpoint(
        args.checkpoint, args.output, verify=not args.no_verify,
    ))


if __name__ == "__main__":
    main()
