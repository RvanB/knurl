"""Train and validate a byte-level BiGRU syntax highlighter."""

from __future__ import annotations

import argparse
import json
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


@dataclass(frozen=True)
class TrainConfig:
    fragment_length: int = 256
    batch_size: int = 32
    train_samples: int = 4096
    validation_samples: int = 512
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    targeted_fraction: float = 0.3
    seed: int = 0
    device: str = "auto"
    log_every_steps: int = 50


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "neural-highlight"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"
    tags: tuple[str, ...] = ()
    log_model: bool = True


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
) -> dict[str, object]:
    if log_every_steps <= 0:
        raise ValueError("log_every_steps must be positive")
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_ID)
    metrics = ClassificationMetrics(model.config.num_classes)
    total_loss = 0.0
    batches = 0
    interval_loss = 0.0
    interval_examples = 0
    interval_bytes = 0
    interval_started = time.perf_counter()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        language_id = batch["language_id"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(input_ids, language_id)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
        total_loss += float(loss.detach())
        batches += 1
        interval_loss += float(loss.detach())
        interval_examples += input_ids.shape[0]
        interval_bytes += int(batch["attention_mask"].sum())
        metrics.update(logits, labels)
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
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    train_config: TrainConfig,
    model_config: BiGRUConfig,
    wandb_config: WandbConfig | None = None,
) -> Path:
    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    train_data = FragmentDataset(
        train_path, train_config.fragment_length, train_config.train_samples,
        train_config.targeted_fraction, train_config.seed,
    )
    validation_data = FragmentDataset(
        validation_path, train_config.fragment_length, train_config.validation_samples,
        targeted_fraction=0.0, seed=train_config.seed + 1,
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    train_loader = DataLoader(train_data, batch_size=train_config.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=train_config.batch_size)
    model = ByteBiGRU(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"device": str(device), "parameters": model.parameter_count}))
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
                "train_path": str(train_path),
                "validation_path": str(validation_path),
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
    best_f1 = -1.0
    best_path = output_dir / "best.pt"
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
        for epoch in range(1, train_config.epochs + 1):
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
            )
            with torch.no_grad():
                validation_metrics = run_epoch(model, validation_loader, device)
            summary = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            print(json.dumps(summary))
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, train_config, validation_metrics)
            improved = float(validation_metrics["macro_f1"]) > best_f1
            if improved:
                best_f1 = float(validation_metrics["macro_f1"])
                save_checkpoint(best_path, model, optimizer, epoch, train_config, validation_metrics)
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        "best/validation_macro_f1": best_f1,
                        "checkpoint/improved": int(improved),
                        **flatten_metrics("train", train_metrics),
                        **flatten_metrics("validation", validation_metrics),
                    },
                    step=global_step,
                )
        if run is not None and tracking.log_model:
            run.log_model(path=str(best_path), name=f"{run.id}-best-bigru")
    finally:
        if run is not None:
            run.finish()
    return best_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/bigru"))
    parser.add_argument("--fragment-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every-steps", type=int, default=50)
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
