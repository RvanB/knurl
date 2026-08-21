from pathlib import Path

import torch

from neural_highlight.challenge import challenge_document, evaluate_challenge
from neural_highlight.dataset.storage import StoredFile, write_record
from neural_highlight.dataset.streams import StreamingFragmentDataset
from neural_highlight.models.streaming_gru import StreamingByteGRU, StreamingGRUConfig
from neural_highlight.labels import Enclosure, LABEL_NAMES, Label
from neural_highlight.languages import LANGUAGE_NAMES
from neural_highlight.stream_infer import stabilize_alphabetic_runs, stream_highlight
from neural_highlight.train_streaming import ExponentialMovingAverage, region_smoothness_loss


def test_streaming_model_carries_state_and_uses_lookahead() -> None:
    model = StreamingByteGRU(StreamingGRUConfig(hidden_size=8, num_layers=1))
    state = model.initial_state(2)
    inputs = torch.randint(0, 256, (2, 24))
    syntax, regions, next_state = model.forward_chunk(inputs, state, 16, torch.tensor([2, 3]))
    assert syntax.shape == (2, 16, len(LABEL_NAMES))
    assert regions.shape == (2, 16, len(LANGUAGE_NAMES))
    assert next_state.shape == (1, 2, 8)
    assert not torch.equal(state, next_state)


def test_streaming_token_context_broadcasts_one_feature_per_word() -> None:
    model = StreamingByteGRU(StreamingGRUConfig(
        hidden_size=8, num_layers=1, token_context_dim=4,
    ))
    inputs = torch.tensor([list(b"This function returns value.")])
    state = model.initial_state(1)
    syntax, _, _ = model.forward_chunk(inputs, state, 20, torch.tensor([0]))
    assert syntax.shape == (1, 20, len(LABEL_NAMES))


def test_streaming_ema_updates_and_temporarily_applies_shadow_weights() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = ExponentialMovingAverage(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(1)
    ema.update(model)
    raw = model.weight.detach().clone()
    assert 0 < float(ema.shadow["weight"].mean()) < 1
    with ema.apply_to(model):
        assert torch.equal(model.weight, ema.shadow["weight"])
    assert torch.equal(model.weight, raw)


def test_region_conditioned_syntax_starts_as_shared_head_and_uses_soft_adapters() -> None:
    config = StreamingGRUConfig(
        hidden_size=8, num_layers=1, region_conditioned_syntax=True,
    )
    model = StreamingByteGRU(config)
    inputs = torch.randint(0, 256, (1, 24))
    state = model.initial_state(1)
    baseline, _, _ = model.forward_chunk(inputs, state, 16, torch.tensor([0]))
    with torch.no_grad():
        model.region_syntax_bias[0, 1] = 4
        model.region_classifier.weight.zero_()
        model.region_classifier.bias.fill_(-10)
        model.region_classifier.bias[0] = 10
    conditioned, _, _ = model.forward_chunk(inputs, state, 16, torch.tensor([0]))
    assert torch.allclose(conditioned[..., 1], baseline[..., 1] + 4, atol=1e-4)
    model.zero_grad(set_to_none=True)
    conditioned.sum().backward()
    assert model.region_classifier.weight.grad is None


def test_region_smoothness_penalizes_jitter_but_not_true_boundaries() -> None:
    stable = torch.tensor([[[8.0, 0.0], [8.0, 0.0], [8.0, 0.0]]])
    jitter = torch.tensor([[[8.0, 0.0], [0.0, 8.0], [8.0, 0.0]]])
    same_targets = torch.tensor([[0, 0, 0]])
    boundary_targets = torch.tensor([[0, 1, 0]])
    assert region_smoothness_loss(stable, same_targets) == 0
    assert region_smoothness_loss(jitter, same_targets) > 1
    assert region_smoothness_loss(jitter, boundary_targets) == 0


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


def test_alphabetic_run_decoder_pools_classes_without_crossing_nonletters() -> None:
    source = b"function x9abc"
    logits = torch.zeros((len(source), len(LABEL_NAMES)))
    logits[:4, 1] = 3
    logits[4:8, 6] = 1
    logits[9, 2] = 4
    logits[10, 10] = 4
    logits[11:, 3] = 4
    labels = stabilize_alphabetic_runs(source, logits)
    assert labels[:8] == bytes([1]) * 8
    assert labels[9] == 2
    assert labels[10] == 10
    assert labels[11:] == bytes([3]) * 3


def test_streaming_challenge_has_fixed_behavioral_slices() -> None:
    document = challenge_document()
    assert len(document.source) == len(document.labels) == len(document.regions)
    assert set(document.slices) == {
        "prose", "embedded_code", "embedded_string_code", "decoration",
        "ordinary_code", "long_string", "language_transition",
    }
    assert all(len(mask) == len(document.source) and any(mask) for mask in document.slices.values())
    long_string_mask = document.slices["long_string"]
    assert sum(long_string_mask) > 3 * 256
    model = StreamingByteGRU(StreamingGRUConfig(hidden_size=8, num_layers=1))
    metrics = evaluate_challenge(model, torch.device("cpu"), chunk_length=32, lookahead=8)
    assert 0 <= metrics["score"] <= 1
    assert all("macro_f1" in metrics[name] for name in document.slices)


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
    inputs = bytes(sample["input_ids"][:, :16].flatten().tolist())
    labels = sample["labels"].flatten().tolist()
    code_start = inputs.index(b"fn live")
    assert labels[code_start : code_start + 7] == [1] * 7


def test_streaming_prose_code_augmentation_labels_prose(tmp_path: Path) -> None:
    path = tmp_path / "prose.jsonl"
    prose = b"This function returns the configured value."
    source = b"/* " + prose + b" */\nreturn value"
    labels = bytes([0]) * (len(source) - 12) + bytes([1]) * 6 + bytes([0]) + bytes([2]) * 5
    mask = bytearray([1]) * len(source)
    mask[3 : 3 + len(prose)] = bytes(len(prose))
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, StoredFile("r", "python", "x.py", source, labels, label_mask=bytes(mask)))
    sample = StreamingFragmentDataset(
        path, chunk_length=32, lookahead=8, chunks_per_sample=2,
        samples_per_epoch=1, delimiter_wrap_fraction=0, prose_code_fraction=1,
    )[0]
    inputs = bytes(sample["input_ids"][:, :32].flatten().tolist())
    targets = sample["labels"].flatten().tolist()
    assert any(word in inputs for word in (b"function", b"returns", b"configured"))
    assert targets.count(Label.PLAIN) >= 32
    assert torch.all(sample["enclosure_labels"] == Enclosure.COMMENT)


def test_streaming_mixed_wrapper_contains_multiple_language_regions(tmp_path: Path) -> None:
    paths = []
    for language, source in (
        ("python", b"def alpha():\n    return 1\n" * 20),
        ("java", b"public int beta() { return 2; }\n" * 20),
        ("rust", b"fn gamma() -> usize { 3 }\n" * 20),
    ):
        path = tmp_path / f"{language}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            write_record(stream, StoredFile("r", language, "x", source, bytes([1]) * len(source)))
        paths.append(path)
    dataset = StreamingFragmentDataset(
        paths, chunk_length=32, lookahead=8, chunks_per_sample=4, samples_per_epoch=16,
        prose_code_fraction=0.125, delimiter_wrap_fraction=0.25,
        mixed_language_fraction=0.125,
    )
    # Two prose and four ordinary wrapped slots precede mixed slots in each cycle.
    sample = dataset[6]
    regions = sample["region_labels"].flatten()
    regions = regions[regions >= 0]
    assert len(torch.unique(regions)) >= 3
    assert int(sample["language_id"]) == 0
    assert torch.any(sample["loss_weights"] == 2)
    assert torch.any(sample["loss_weights"] == 1)


def test_streaming_string_wrapper_preserves_embedded_code_regions(tmp_path: Path) -> None:
    paths = []
    prose = b"This explanation describes a configured shader value."
    for language, source in (
        ("python", b"def host():\n    return 1\n" * 20),
        ("java", b"public int payload() { return 2; }\n" * 20),
    ):
        path = tmp_path / f"{language}.jsonl"
        combined = prose + b"\n" + source
        labels = bytes([Label.COMMENT]) * len(prose) + b"\x00" + bytes([1]) * len(source)
        with path.open("w", encoding="utf-8") as stream:
            write_record(stream, StoredFile("r", language, "x", combined, labels))
        paths.append(path)
    dataset = StreamingFragmentDataset(
        paths, chunk_length=32, lookahead=8, chunks_per_sample=4, samples_per_epoch=16,
        prose_code_fraction=0.125, delimiter_wrap_fraction=0.25,
        mixed_language_fraction=0.125, string_code_fraction=0.125,
    )
    sample = dataset[8]
    regions = sample["region_labels"].flatten()
    regions = regions[regions >= 0]
    assert len(torch.unique(regions)) >= 2
    assert torch.any(sample["loss_weights"] == 2)
    assert torch.any(sample["enclosure_labels"] == Enclosure.STRING)
    assert int((sample["labels"] == Label.PLAIN).sum()) > 20  # Prose inside the literal.


def test_streaming_long_string_replay_labels_entire_literal_as_string(tmp_path: Path) -> None:
    path = tmp_path / "python.jsonl"
    source = b"description = 'ordinary text'\n" * 20
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, StoredFile("r", "python", "x.py", source, bytes([0]) * len(source)))
    sample = StreamingFragmentDataset(
        path, chunk_length=32, lookahead=8, chunks_per_sample=4,
        samples_per_epoch=16, prose_code_fraction=0, delimiter_wrap_fraction=0,
        long_string_fraction=1,
    )[0]
    targets = sample["labels"].flatten()
    assert torch.all(targets == Label.PLAIN)
    assert torch.all(sample["enclosure_labels"] == Enclosure.STRING)
    regions = sample["region_labels"].flatten()
    assert torch.all(regions == LANGUAGE_NAMES.index("python"))


def test_balanced_replay_assigns_each_category_within_one_batch(tmp_path: Path) -> None:
    path = tmp_path / "python.jsonl"
    source = b"x = 1\n" * 100
    with path.open("w", encoding="utf-8") as stream:
        write_record(stream, StoredFile("r", "python", "x.py", source, bytes([0]) * len(source)))
    dataset = StreamingFragmentDataset(
        path, samples_per_epoch=16, quota_cycle=16,
        prose_code_fraction=0.125, delimiter_wrap_fraction=0.125,
        mixed_language_fraction=0.125, string_code_fraction=0.125,
        long_string_fraction=0.125, mixed_ordinary_fraction=0.125,
    )
    kinds = [dataset._sample_kind(index) for index in range(16)]
    assert kinds.count("ordinary") == 4
    for kind in (
        "prose_code", "wrapped_code", "mixed_wrapped", "string_code",
        "long_string", "mixed_ordinary",
    ):
        assert kinds.count(kind) == 2


def test_streaming_unwrapped_mixture_switches_regions_with_unknown_host(tmp_path: Path) -> None:
    paths = []
    for language, source in (
        ("python", b"def alpha():\n    return 1\n" * 20),
        ("javascript", b"function beta() { return 2; }\n" * 20),
        ("java", b"public int gamma() { return 3; }\n" * 20),
    ):
        path = tmp_path / f"{language}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            write_record(stream, StoredFile("r", language, "x", source, bytes([1]) * len(source)))
        paths.append(path)
    dataset = StreamingFragmentDataset(
        paths, chunk_length=32, lookahead=8, chunks_per_sample=4, samples_per_epoch=16,
        prose_code_fraction=0.125, delimiter_wrap_fraction=0.25,
        mixed_language_fraction=0.125, string_code_fraction=0.125,
        mixed_ordinary_fraction=0.125,
    )
    sample = dataset[10]
    regions = sample["region_labels"].flatten()
    regions = regions[regions >= 0]
    assert len(torch.unique(regions)) >= 3
    assert int(sample["language_id"]) == 0
    assert torch.all(sample["loss_weights"] == 1)
