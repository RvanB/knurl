"""Train the persistent-state GRU over consecutive source chunks."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neural_highlight.dataset.fragments import IGNORE_LABEL_ID
from neural_highlight.dataset.streams import StreamingFragmentDataset
from neural_highlight.languages import LANGUAGE_NAMES
from neural_highlight.metrics import ClassificationMetrics
from neural_highlight.models.streaming_gru import StreamingByteGRU, StreamingGRUConfig
from neural_highlight.train import EarlyStopping, resolve_device, set_seed


@dataclass(frozen=True)
class StreamingTrainConfig:
    chunk_length: int = 256
    lookahead: int = 128
    chunks_per_sample: int = 8
    batch_size: int = 16
    train_samples: int = 4096
    validation_samples: int = 512
    epochs: int = 0
    learning_rate: float = 1e-3
    region_loss_weight: float = 0.2
    host_hint_dropout: float = 0.5
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    seed: int = 0
    device: str = "auto"
    log_every_steps: int = 20
    num_workers: int = 2
    delimiter_wrap_fraction: float = 0.1


def run_stream_epoch(
    model: StreamingByteGRU,
    loader: DataLoader,
    device: torch.device,
    config: StreamingTrainConfig,
    optimizer: AdamW | None = None,
    step_callback=None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_ID)
    syntax_metrics = ClassificationMetrics(model.config.num_classes)
    region_metrics = ClassificationMetrics(model.config.num_languages, LANGUAGE_NAMES)
    loss_total = syntax_total = region_total = 0.0
    for batch_index, batch in enumerate(loader, 1):
        non_blocking = device.type == "cuda"
        inputs = batch["input_ids"].to(device, non_blocking=non_blocking)
        labels = batch["labels"].to(device, non_blocking=non_blocking)
        regions = batch["region_labels"].to(device, non_blocking=non_blocking)
        language = batch["language_id"].to(device, non_blocking=non_blocking)
        if training and config.host_hint_dropout:
            language = language.masked_fill(
                torch.rand(language.shape, device=device) < config.host_hint_dropout, 0
            )
        state = model.initial_state(inputs.shape[0], device)
        chunk_losses = []
        syntax_losses = []
        region_losses = []
        with torch.set_grad_enabled(training):
            for chunk_index in range(inputs.shape[1]):
                syntax_logits, region_logits, state = model.forward_chunk(
                    inputs[:, chunk_index], state, config.chunk_length, language
                )
                if not torch.any(labels[:, chunk_index] != IGNORE_LABEL_ID):
                    continue
                syntax_loss = criterion(
                    syntax_logits.reshape(-1, syntax_logits.shape[-1]),
                    labels[:, chunk_index].reshape(-1),
                )
                region_loss = criterion(
                    region_logits.reshape(-1, region_logits.shape[-1]),
                    regions[:, chunk_index].reshape(-1),
                )
                syntax_losses.append(syntax_loss)
                region_losses.append(region_loss)
                chunk_losses.append(syntax_loss + config.region_loss_weight * region_loss)
                syntax_metrics.update(syntax_logits, labels[:, chunk_index])
                region_metrics.update(region_logits, regions[:, chunk_index])
            loss = torch.stack(chunk_losses).mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
        syntax_mean = torch.stack(syntax_losses).mean()
        region_mean = torch.stack(region_losses).mean()
        loss_total += float(loss.detach())
        syntax_total += float(syntax_mean.detach())
        region_total += float(region_mean.detach())
        if training and step_callback is not None and (
            batch_index % config.log_every_steps == 0 or batch_index == len(loader)
        ):
            step_callback(
                {
                    "batch": batch_index,
                    "loss": float(loss.detach()),
                    "gradient_norm": gradient_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    result = syntax_metrics.compute()
    batches = max(1, len(loader))
    result.update(
        {
            "loss": loss_total / batches,
            "syntax_loss": syntax_total / batches,
            "region_loss": region_total / batches,
            "region": region_metrics.compute(),
        }
    )
    return result


def train_streaming(
    train_paths: list[Path],
    validation_paths: list[Path],
    output: Path,
    config: StreamingTrainConfig,
    model_config: StreamingGRUConfig,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> Path:
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_data = StreamingFragmentDataset(
        train_paths, config.chunk_length, config.lookahead, config.chunks_per_sample,
        config.train_samples, config.seed,
        delimiter_wrap_fraction=config.delimiter_wrap_fraction,
    )
    validation_data = StreamingFragmentDataset(
        validation_paths, config.chunk_length, config.lookahead, config.chunks_per_sample,
        config.validation_samples, config.seed + 1,
        delimiter_wrap_fraction=0.0,
    )
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_kwargs)
    model = StreamingByteGRU(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"device": str(device), "parameters": model.parameter_count, **asdict(config)}))
    run = None
    if wandb_project:
        import wandb

        run = wandb.init(
            project=wandb_project, name=wandb_name, job_type="train-streaming",
            dir=str(output), config={**asdict(config), **model_config.to_dict()},
        )
    stopper = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)
    best_path = output / "best.pt"
    global_step = 0

    def log_step(values):
        nonlocal global_step
        global_step += config.log_every_steps
        payload = {"global_step": global_step, **{f"train_step/{k}": v for k, v in values.items()}}
        print(json.dumps(payload))
        if run:
            run.log(payload, step=global_step)

    try:
        epochs = itertools.count(1) if config.epochs <= 0 else range(1, config.epochs + 1)
        for epoch in epochs:
            train_data.set_epoch(epoch)
            validation_data.set_epoch(epoch)
            train_metrics = run_stream_epoch(model, train_loader, device, config, optimizer, log_step)
            with torch.no_grad():
                validation_metrics = run_stream_epoch(model, validation_loader, device, config)
            improved, should_stop, _ = stopper.update(float(validation_metrics["accuracy"]))
            checkpoint = {
                "architecture": "streaming-gru",
                "model_config": model.config.to_dict(),
                "train_config": asdict(config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": validation_metrics,
            }
            torch.save(checkpoint, output / "last.pt")
            if improved:
                torch.save(checkpoint, best_path)
            summary = {
                "epoch": epoch, "train": train_metrics, "validation": validation_metrics,
                "early_stopping": {"best": stopper.best, "bad_epochs": stopper.bad_epochs},
            }
            print(json.dumps(summary))
            if run:
                run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/accuracy": train_metrics["accuracy"],
                        "validation/loss": validation_metrics["loss"],
                        "validation/accuracy": validation_metrics["accuracy"],
                        "validation/macro_f1": validation_metrics["macro_f1"],
                    },
                    step=global_step,
                )
            if should_stop:
                break
        if run:
            run.log_model(str(best_path), name=f"{run.id}-streaming-gru")
    finally:
        if run:
            run.finish()
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/streaming-gru"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--lookahead", type=int, default=128)
    parser.add_argument("--chunks-per-sample", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--delimiter-wrap-fraction", type=float, default=0.1)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    args = parser.parse_args()
    config = StreamingTrainConfig(
        chunk_length=args.chunk_length, lookahead=args.lookahead,
        chunks_per_sample=args.chunks_per_sample, batch_size=args.batch_size,
        train_samples=args.train_samples, validation_samples=args.validation_samples,
        epochs=args.epochs, early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate, num_workers=args.num_workers,
        log_every_steps=args.log_every_steps, device=args.device,
        delimiter_wrap_fraction=args.delimiter_wrap_fraction,
    )
    print(train_streaming(
        args.train, args.validation, args.output, config,
        StreamingGRUConfig(), args.wandb_project, args.wandb_name,
    ))


if __name__ == "__main__":
    main()
