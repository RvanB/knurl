import torch

from neural_highlight.dataset.fragments import IGNORE_LABEL_ID, PAD_BYTE_ID
from neural_highlight.infer import highlight
from neural_highlight.metrics import ClassificationMetrics
from neural_highlight.models.bigru import BiGRUConfig, ByteBiGRU
from neural_highlight.train import flatten_metrics


def test_bigru_shape_and_language_modes() -> None:
    inputs = torch.tensor([[100, 101, 102, PAD_BYTE_ID], [1, 2, 3, 4]])
    languages = torch.tensor([1, 0])
    for use_language in (False, True):
        model = ByteBiGRU(BiGRUConfig(hidden_size=8, num_layers=1, use_language_embedding=use_language))
        assert model(inputs, languages).shape == (2, 4, 14)
        assert model.parameter_count > 0


def test_metrics_ignore_padding_and_are_exact() -> None:
    logits = torch.zeros(1, 3, 14)
    logits[0, 0, 1] = 1
    logits[0, 1, 2] = 1
    targets = torch.tensor([[1, 2, IGNORE_LABEL_ID]])
    metrics = ClassificationMetrics()
    metrics.update(logits, targets)
    result = metrics.compute()
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_inference_returns_one_label_per_byte() -> None:
    model = ByteBiGRU(BiGRUConfig(hidden_size=8, num_layers=1))
    source = "café()".encode()
    assert len(highlight(model, source, "python")) == len(source)
    assert highlight(model, b"") == b""


def test_wandb_metrics_are_flattened() -> None:
    metrics = {
        "loss": 1.5,
        "accuracy": 0.5,
        "macro_f1": 0.25,
        "per_class": {
            "keyword": {"precision": 0.4, "recall": 0.3, "f1": 0.34, "support": 12}
        },
    }
    flat = flatten_metrics("validation", metrics)
    assert flat["validation/loss"] == 1.5
    assert flat["validation/class/keyword/f1"] == 0.34
    assert flat["validation/class/keyword/support"] == 12
