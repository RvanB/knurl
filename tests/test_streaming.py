from pathlib import Path

import torch

from neural_highlight.dataset.storage import StoredFile, write_record
from neural_highlight.dataset.streams import StreamingFragmentDataset
from neural_highlight.models.streaming_gru import StreamingByteGRU, StreamingGRUConfig
from neural_highlight.stream_infer import stream_highlight


def test_streaming_model_carries_state_and_uses_lookahead() -> None:
    model = StreamingByteGRU(StreamingGRUConfig(hidden_size=8, num_layers=1))
    state = model.initial_state(2)
    inputs = torch.randint(0, 256, (2, 24))
    syntax, regions, next_state = model.forward_chunk(inputs, state, 16, torch.tensor([2, 3]))
    assert syntax.shape == (2, 16, 14)
    assert regions.shape == (2, 16, 17)
    assert next_state.shape == (1, 2, 8)
    assert not torch.equal(state, next_state)


def test_streaming_dataset_preserves_consecutive_chunks(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    source = bytes(range(100))
    record = StoredFile("r", "python", "x.py", source, bytes([2]) * len(source))
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, record)
    dataset = StreamingFragmentDataset(
        path, chunk_length=8, lookahead=4, chunks_per_sample=3,
        samples_per_epoch=1, delimiter_wrap_fraction=0,
    )
    sample = dataset[0]
    assert sample["input_ids"].shape == (3, 12)
    assert torch.equal(sample["input_ids"][0, 8:12], sample["input_ids"][1, :4])
    assert torch.equal(sample["input_ids"][1, 8:12], sample["input_ids"][2, :4])


def test_streaming_inference_returns_checkpoints_for_long_input() -> None:
    model = StreamingByteGRU(StreamingGRUConfig(hidden_size=8, num_layers=1))
    source = b"/*" + b"fn main() {}\n" * 20 + b"*/"
    prediction = stream_highlight(model, source, "rust", chunk_length=32, lookahead=8)
    assert len(prediction.syntax) == len(source)
    assert len(prediction.regions) == len(source)
    assert len(prediction.checkpoints) == (len(source) + 31) // 32 + 1


def test_streaming_wrapped_code_keeps_interior_syntax_labels(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    source = b"fn live() {}" * 10
    labels = bytes([1]) * len(source)
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, StoredFile("r", "rust", "x.rs", source, labels))
    dataset = StreamingFragmentDataset(
        path, chunk_length=16, lookahead=4, chunks_per_sample=3,
        samples_per_epoch=1, delimiter_wrap_fraction=1,
    )
    sample = dataset[0]
    assert sample["input_ids"][0, :2].tolist() == list(b"/*")
    assert sample["labels"][0, 2:8].tolist() == [1] * 6
