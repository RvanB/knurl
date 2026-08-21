import io
import json

import torch

from neural_highlight.stream_infer import StreamPrediction
from neural_highlight.stream_server import prediction_spans, serve


def test_prediction_spans_merges_equal_syntax_and_language() -> None:
    prediction = StreamPrediction(
        syntax=bytes([1, 1, 2, 2]),
        regions=bytes([3, 3, 3, 4]),
        checkpoints=(torch.zeros(1),),
    )
    assert prediction_spans(prediction) == [
        [0, 2, 1, 3], [2, 3, 2, 3], [3, 4, 2, 4],
    ]


def test_stream_server_returns_one_json_response_per_request(monkeypatch) -> None:
    def predict(model, source: bytes, language: str) -> StreamPrediction:
        assert language == "unknown"
        return StreamPrediction(
            syntax=bytes([1]) * len(source), regions=bytes([0]) * len(source),
            checkpoints=(),
        )

    monkeypatch.setattr("neural_highlight.stream_server.stream_highlight_onnx", predict)
    output = io.StringIO()
    serve(object(), io.StringIO('{"id":7,"text":"hello"}\nnot-json\n'), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]["id"] == 7
    assert responses[0]["spans"] == [[0, 5, 1, 0]]
    assert responses[1]["id"] is None
    assert "error" in responses[1]
