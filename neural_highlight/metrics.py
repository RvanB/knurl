"""Dependency-free multiclass metrics accumulated as a confusion matrix."""

from __future__ import annotations

import torch
from torch import Tensor

from neural_highlight.dataset.fragments import IGNORE_LABEL_ID
from neural_highlight.labels import LABEL_NAMES


class ClassificationMetrics:
    def __init__(self, num_classes: int = len(LABEL_NAMES), class_names: tuple[str, ...] = LABEL_NAMES) -> None:
        self.num_classes = num_classes
        self.class_names = class_names
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def update(self, logits: Tensor, targets: Tensor) -> None:
        predictions = logits.argmax(dim=-1).detach().cpu().reshape(-1)
        targets = targets.detach().cpu().reshape(-1)
        valid = targets != IGNORE_LABEL_ID
        indices = targets[valid] * self.num_classes + predictions[valid]
        self.confusion += torch.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, object]:
        matrix = self.confusion.float()
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        precision = true_positive / predicted.clamp_min(1)
        recall = true_positive / support.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        total = support.sum()
        per_class = {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(self.class_names)
        }
        present = support > 0
        return {
            "accuracy": float(true_positive.sum() / total.clamp_min(1)),
            "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "per_class": per_class,
        }
