"""Train and validate a byte-level BiGRU syntax highlighter."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neural_highlight.dataset.fragments import FragmentDataset, IGNORE_LABEL_ID
from neural_highlight.metrics import ClassificationMetrics
from neural_highlight.models.bigru import BiGRUConfig, ByteBiGRU
from neural_highlight.languages import LANGUAGE_NAMES


@dataclass(frozen=True)
class TrainConfig:
    fragment_length: int = 256
    batch_size: int = 32
    train_samples: int = 4096
    validation_samples: int = 512
    epochs: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    targeted_fraction: float = 0.3
    seed: int = 0
    device: str = "auto"
    log_every_steps: int = 50
    mixture_fraction: float = 0.25
    max_regions: int = 4
    region_loss_weight: float = 0.2
    host_hint_dropout: float = 0.5
    early_stopping_patience: int = 3
    early_stopping_metric: str = "accuracy"
    early_stopping_min_delta: float = 0.0
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "neural-highlight"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"
    tags: tuple[str, ...] = ()
    log_model: bool = True


class EarlyStopping:
    """Track meaningful validation improvements and patience exhaustion."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 0:
            raise ValueError("early-stopping patience cannot be negative")
        if min_delta < 0:
            raise ValueError("early-stopping min delta cannot be negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.bad_epochs = 0
        self._needs_after_best = False

    def update(self, value: float) -> tuple[bool, bool, bool]:
        """Return ``(improved, should_stop, save_after_best)``."""
        improved = math.isfinite(value) and value > self.best + self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
            self._needs_after_best = True
            return True, False, False
        self.bad_epochs += 1
        save_after_best = self._needs_after_best
        self._needs_after_best = False
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return False, should_stop, save_after_best


def flatten_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, float | int]:
    """Convert nested metrics into stable W&B chart keys."""
    result: dict[str, float | int] = {
        f"{prefix}/loss": float(metrics["loss"]),
        f"{prefix}/accuracy": float(metrics["accuracy"]),
        f"{prefix}/macro_f1": float(metrics["macro_f1"]),
    }
    per_class = metrics["per_class"]
    assert isinstance(per_class, dict)
    for class_name, values in per_class.items():
        assert isinstance(values, dict)
        result[f"{prefix}/class/{class_name}/precision"] = float(values["precision"])
        result[f"{prefix}/class/{class_name}/recall"] = float(values["recall"])
        result[f"{prefix}/class/{class_name}/f1"] = float(values["f1"])
        result[f"{prefix}/class/{class_name}/support"] = int(values["support"])
    region = metrics.get("region")
    if isinstance(region, dict):
        result[f"{prefix}/region/accuracy"] = float(region["accuracy"])
        result[f"{prefix}/region/macro_f1"] = float(region["macro_f1"])
        region_classes = region["per_class"]
        assert isinstance(region_classes, dict)
        for class_name, values in region_classes.items():
            assert isinstance(values, dict)
            result[f"{prefix}/region/class/{class_name}/f1"] = float(values["f1"])
            result[f"{prefix}/region/class/{class_name}/support"] = int(values["support"])
    result[f"{prefix}/syntax_loss"] = float(metrics.get("syntax_loss", metrics["loss"]))
    result[f"{prefix}/region_loss"] = float(metrics.get("region_loss", 0.0))
    return result


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: ByteBiGRU,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None = None,
    step_callback: Callable[[dict[str, float | int]], None] | None = None,
    log_every_steps: int = 50,
    region_loss_weight: float = 0.2,
    host_hint_dropout: float = 0.0,
) -> dict[str, object]:
    if log_every_steps <= 0:
        raise ValueError("log_every_steps must be positive")
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_ID)
    metrics = ClassificationMetrics(model.config.num_classes)
    region_metrics = ClassificationMetrics(model.config.num_languages, LANGUAGE_NAMES)
    total_loss = 0.0
    total_syntax_loss = 0.0
    total_region_loss = 0.0
    batches = 0
    interval_loss = 0.0
    interval_examples = 0
    interval_bytes = 0
    interval_started = time.perf_counter()
    for batch in loader:
        non_blocking = device.type == "cuda"
        input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
        labels = batch["labels"].to(device, non_blocking=non_blocking)
        region_labels = batch["region_labels"].to(device, non_blocking=non_blocking)
        language_id = batch["language_id"].to(device, non_blocking=non_blocking)
        if training and host_hint_dropout > 0:
            dropped = torch.rand(language_id.shape, device=device) < host_hint_dropout
            language_id = language_id.masked_fill(dropped, 0)
        with torch.set_grad_enabled(training):
            logits, region_logits = model.forward_with_regions(input_ids, language_id)
            syntax_loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            region_loss = criterion(
                region_logits.reshape(-1, region_logits.shape[-1]), region_labels.reshape(-1)
            )
            loss = syntax_loss + region_loss_weight * region_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
        total_loss += float(loss.detach())
        total_syntax_loss += float(syntax_loss.detach())
        total_region_loss += float(region_loss.detach())
        batches += 1
        interval_loss += float(loss.detach())
        interval_examples += input_ids.shape[0]
        interval_bytes += int(batch["attention_mask"].sum())
        metrics.update(logits, labels)
        region_metrics.update(region_logits, region_labels)
        if training and step_callback is not None and (
            batches % log_every_steps == 0 or batches == len(loader)
        ):
            elapsed = max(time.perf_counter() - interval_started, 1e-9)
            step_values: dict[str, float | int] = {
                "batch": batches,
                "loss": float(loss.detach()),
                "rolling_loss": interval_loss / max(1, batches % log_every_steps or log_every_steps),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm": gradient_norm,
                "examples_per_second": interval_examples / elapsed,
                "bytes_per_second": interval_bytes / elapsed,
            }
            if device.type == "cuda":
                step_values.update(
                    {
                        "cuda_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
                        "cuda_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
                        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                    }
                )
            step_callback(step_values)
            interval_loss = 0.0
            interval_examples = 0
            interval_bytes = 0
            interval_started = time.perf_counter()
    result = metrics.compute()
    result["loss"] = total_loss / max(1, batches)
    result["syntax_loss"] = total_syntax_loss / max(1, batches)
    result["region_loss"] = total_region_loss / max(1, batches)
    result["region"] = region_metrics.compute()
    return result


def save_checkpoint(
    path: Path,
    model: ByteBiGRU,
    optimizer: AdamW,
    epoch: int,
    train_config: TrainConfig,
    metrics: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.config.to_dict(),
            "train_config": asdict(train_config),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def train(
    train_path: Path | list[Path],
    validation_path: Path | list[Path],
    output_dir: Path,
    train_config: TrainConfig,
    model_config: BiGRUConfig,
    wandb_config: WandbConfig | None = None,
) -> Path:
    if train_config.epochs <= 0 and train_config.early_stopping_patience <= 0:
        raise ValueError("unlimited epochs require positive early-stopping patience")
    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    train_data = FragmentDataset(
        train_path, train_config.fragment_length, train_config.train_samples,
        train_config.targeted_fraction, train_config.seed,
        train_config.mixture_fraction, train_config.max_regions,
    )
    validation_data = FragmentDataset(
        validation_path, train_config.fragment_length, train_config.validation_samples,
        targeted_fraction=0.0, seed=train_config.seed + 1,
        mixture_fraction=train_config.mixture_fraction,
        max_regions=train_config.max_regions,
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    if train_config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if train_config.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    loader_options: dict[str, object] = {
        "batch_size": train_config.batch_size,
        "num_workers": train_config.num_workers,
        "pin_memory": train_config.pin_memory and device.type == "cuda",
    }
    if train_config.num_workers > 0:
        persistent_workers = (
            train_config.persistent_workers
            and train_data.epoch_is_shared
            and validation_data.epoch_is_shared
        )
        loader_options.update(
            {
                "prefetch_factor": train_config.prefetch_factor,
                "persistent_workers": persistent_workers,
            }
        )
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    model = ByteBiGRU(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "device": str(device),
        "parameters": model.parameter_count,
        "dataloader": {
            "num_workers": train_config.num_workers,
            "prefetch_factor": train_config.prefetch_factor if train_config.num_workers else None,
            "pin_memory": bool(loader_options["pin_memory"]),
            "persistent_workers": bool(loader_options.get("persistent_workers", False)),
            "persistent_workers_requested": train_config.persistent_workers,
        },
    }))
    tracking = wandb_config or WandbConfig()
    run = None
    if tracking.enabled:
        import wandb

        run = wandb.init(
            project=tracking.project,
            entity=tracking.entity,
            name=tracking.name,
            mode=tracking.mode,
            tags=list(tracking.tags),
            job_type="train",
            dir=str(output_dir),
            config={
                "architecture": "byte-bigru",
                "parameters": model.parameter_count,
                "train_path": [str(path) for path in train_path] if isinstance(train_path, list) else str(train_path),
                "validation_path": [str(path) for path in validation_path] if isinstance(validation_path, list) else str(validation_path),
                **{f"train/{key}": value for key, value in asdict(train_config).items()},
                **{f"model/{key}": value for key, value in model_config.to_dict().items()},
            },
        )
        run.define_metric("epoch")
        run.define_metric("global_step")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("validation/*", step_metric="epoch")
        run.define_metric("train_step/*", step_metric="global_step")
        run.watch(model, log="gradients", log_freq=max(1, train_config.log_every_steps))
    best_path = output_dir / "best.pt"
    after_best_path = output_dir / "after-best.pt"
    stopper = EarlyStopping(
        train_config.early_stopping_patience, train_config.early_stopping_min_delta
    )
    global_step = 0

    def log_training_step(values: dict[str, float | int]) -> None:
        nonlocal global_step
        # The callback fires after an interval, so advance by the actual number
        # of batches since the last report (the final interval may be shorter).
        current_batch = int(values["batch"])
        previous_batch = global_step % len(train_loader)
        advanced = current_batch - previous_batch
        if advanced <= 0:
            advanced = current_batch
        global_step += advanced
        payload = {
            "global_step": global_step,
            **{f"train_step/{key}": value for key, value in values.items()},
        }
        print(json.dumps(payload))
        if run is not None:
            run.log(payload, step=global_step)

    try:
        epochs = (
            itertools.count(1)
            if train_config.epochs <= 0
            else range(1, train_config.epochs + 1)
        )
        for epoch in epochs:
            train_data.set_epoch(epoch)
            validation_data.set_epoch(epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                step_callback=log_training_step,
                log_every_steps=train_config.log_every_steps,
                region_loss_weight=train_config.region_loss_weight,
                host_hint_dropout=train_config.host_hint_dropout,
            )
            with torch.no_grad():
                validation_metrics = run_epoch(
                    model, validation_loader, device,
                    region_loss_weight=train_config.region_loss_weight,
                )
            summary = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            print(json.dumps(summary))
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, train_config, validation_metrics)
            monitored_value = float(validation_metrics[train_config.early_stopping_metric])
            improved, should_stop, save_after_best = stopper.update(monitored_value)
            if improved:
                save_checkpoint(best_path, model, optimizer, epoch, train_config, validation_metrics)
            elif save_after_best:
                save_checkpoint(
                    after_best_path, model, optimizer, epoch, train_config, validation_metrics
                )
            stopping = {
                "metric": train_config.early_stopping_metric,
                "value": monitored_value,
                "best": stopper.best,
                "bad_epochs": stopper.bad_epochs,
                "patience": stopper.patience,
                "improved": improved,
                "should_stop": should_stop,
            }
            print(json.dumps({"epoch": epoch, "early_stopping": stopping}))
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        f"best/validation_{train_config.early_stopping_metric}": stopper.best,
                        "checkpoint/improved": int(improved),
                        "checkpoint/saved_after_best": int(save_after_best),
                        "early_stopping/bad_epochs": stopper.bad_epochs,
                        "early_stopping/patience": stopper.patience,
                        **flatten_metrics("train", train_metrics),
                        **flatten_metrics("validation", validation_metrics),
                    },
                    step=global_step,
                )
            if should_stop:
                print(json.dumps({"stopped_early": True, "epoch": epoch, **stopping}))
                break
        if run is not None and tracking.log_model:
            run.log_model(path=str(best_path), name=f"{run.id}-best-bigru")
    finally:
        if run is not None:
            run.finish()
    return best_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/bigru"))
    parser.add_argument("--fragment-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="maximum epochs; 0 means unlimited and relies on early stopping",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument(
        "--early-stopping-metric", choices=("accuracy", "macro_f1"), default="accuracy"
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--mixture-fraction", type=float, default=0.25)
    parser.add_argument("--max-regions", type=int, default=4)
    parser.add_argument("--region-loss-weight", type=float, default=0.2)
    parser.add_argument("--host-hint-dropout", type=float, default=0.5)
    parser.add_argument("--language-embedding", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--wandb", action="store_true", help="stream metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="neural-highlight")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-tags", default="", help="comma-separated run tags")
    parser.add_argument("--no-wandb-model", action="store_true", help="do not upload the best checkpoint")
    args = parser.parse_args(argv)
    config = TrainConfig(
        fragment_length=args.fragment_length, batch_size=args.batch_size,
        train_samples=args.train_samples, validation_samples=args.validation_samples,
        epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed, device=args.device,
        log_every_steps=args.log_every_steps,
        mixture_fraction=args.mixture_fraction, max_regions=args.max_regions,
        region_loss_weight=args.region_loss_weight, host_hint_dropout=args.host_hint_dropout,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_min_delta=args.early_stopping_min_delta,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent_workers,
    )
    model_config = BiGRUConfig(
        hidden_size=args.hidden_size, num_layers=args.layers,
        use_language_embedding=args.language_embedding,
    )
    tracking = WandbConfig(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        tags=tuple(tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()),
        log_model=not args.no_wandb_model,
    )
    print(train(args.train, args.validation, args.output, config, model_config, tracking))


if __name__ == "__main__":
    main()
