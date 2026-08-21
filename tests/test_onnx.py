from pathlib import Path

import torch

from neural_highlight.export_onnx import export_streaming_checkpoint
from neural_highlight.labels import LABEL_SCHEMA_VERSION
from neural_highlight.languages import LANGUAGE_SCHEMA_VERSION
from neural_highlight.models.streaming_gru import StreamingByteGRU, StreamingGRUConfig
from neural_highlight.stream_infer import (
    OnnxStreamingSession,
    main as stream_infer_main,
    stream_highlight,
    stream_highlight_onnx,
)


def test_onnx_export_matches_two_chunk_pytorch_inference(tmp_path: Path) -> None:
    config = StreamingGRUConfig(
        byte_embedding_dim=8, hidden_size=8, num_layers=1,
        region_conditioned_syntax=True, token_context_dim=4,
    )
    model = StreamingByteGRU(config).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": "streaming-gru",
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "language_schema_version": LANGUAGE_SCHEMA_VERSION,
            "model_config": config.to_dict(),
            "train_config": {"chunk_length": 16, "lookahead": 8},
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    exported = export_streaming_checkpoint(checkpoint)
    assert exported == checkpoint.with_suffix(".onnx")
    assert exported.is_file()
    onnx_model = OnnxStreamingSession(exported)
    assert onnx_model.chunk_length == 16
    assert onnx_model.lookahead == 8
    source = b"function alpha() { return beta_value; }\n" * 2
    expected = stream_highlight(model, source, "javascript", 16, 8)
    actual = stream_highlight_onnx(onnx_model, source, "javascript")
    assert actual.syntax == expected.syntax
    assert actual.regions == expected.regions
    stream_infer_main([
        str(exported), "--text", "function alpha() {}", "--language", "javascript",
    ])
