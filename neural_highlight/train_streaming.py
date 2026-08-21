"""Train the persistent-state GRU over consecutive source chunks."""

from __future__ import annotations

import argparse
import itertools
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from neural_highlight.dataset.fragments import IGNORE_LABEL_ID
from neural_highlight.dataset.streams import SAMPLE_KINDS, StreamingFragmentDataset
from neural_highlight.challenge import evaluate_challenge
from neural_highlight.languages import LANGUAGE_NAMES, LANGUAGE_SCHEMA_VERSION
from neural_highlight.labels import ENCLOSURE_NAMES, LABEL_SCHEMA_VERSION
from neural_highlight.metrics import ClassificationMetrics
from neural_highlight.models.streaming_gru import StreamingByteGRU, StreamingGRUConfig
from neural_highlight.train import EarlyStopping, resolve_device, set_seed
from neural_highlight.export_onnx import export_streaming_checkpoint


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
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    seed: int = 0
    device: str = "auto"
    log_every_steps: int = 20
    num_workers: int = 2
    delimiter_wrap_fraction: float = 0.1
    prose_code_fraction: float = 0.15
    mixed_language_fraction: float = 0.0
    string_code_fraction: float = 0.0
    long_string_fraction: float = 0.0
    mixed_ordinary_fraction: float = 0.0
    embedded_loss_weight: float = 2.0
    region_smoothness_weight: float = 0.03
    probe_every_steps: int = 250
    validation_every_steps: int = 1000
    minimum_training_steps: int = 20000
    selection_smoothing_window: int = 5
    ema_decay: float = 0.999
    lr_plateau_patience: int = 4
    lr_decay_factor: float = 0.5
    minimum_learning_rate: float = 1e-5
    minimum_ordinary_score: float = 0.90
    minimum_embedded_score: float = 0.75
    minimum_string_embedded_score: float = 0.75
    minimum_long_string_accuracy: float = 0.95
    long_string_loss_scale: float = 0.5
    category_gradient_every_steps: int = 250
    enclosure_loss_weight: float = 0.1


class ExponentialMovingAverage:
    """Maintain inference weights without disturbing optimizer weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Warm up the average so early evaluations do not remain dominated by
        # random initialization; it approaches the configured decay smoothly.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1 - decay)
            else:
                self.shadow[name].copy_(value)

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        original = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(original)

    def state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}


def region_smoothness_loss(
    logits: Tensor,
    targets: Tensor,
    previous_logits: Tensor | None = None,
    previous_targets: Tensor | None = None,
) -> Tensor:
    """Penalize distribution jumps only where the true region stays constant."""
    probabilities = logits.softmax(dim=-1)
    losses: list[Tensor] = []
    if logits.shape[1] > 1:
        same = (
            (targets[:, 1:] == targets[:, :-1])
            & (targets[:, 1:] != IGNORE_LABEL_ID)
        )
        if same.any():
            difference = probabilities[:, 1:] - probabilities[:, :-1]
            losses.append(difference.square().sum(dim=-1)[same])
    if previous_logits is not None and previous_targets is not None:
        same = (
            (targets[:, 0] == previous_targets[:, -1])
            & (targets[:, 0] != IGNORE_LABEL_ID)
        )
        if same.any():
            difference = probabilities[:, 0] - previous_logits[:, -1].softmax(dim=-1)
            losses.append(difference.square().sum(dim=-1)[same])
    if not losses:
        return logits.sum() * 0
    return torch.cat(losses).mean()


def run_stream_epoch(
    model: StreamingByteGRU,
    loader: DataLoader,
    device: torch.device,
    config: StreamingTrainConfig,
    optimizer: AdamW | None = None,
    step_callback=None,
    after_optimizer_step: Callable[[], bool] | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_ID, reduction="none")
    syntax_metrics = ClassificationMetrics(model.config.num_classes)
    region_metrics = ClassificationMetrics(model.config.num_languages, LANGUAGE_NAMES)
    enclosure_metrics = ClassificationMetrics(model.config.num_enclosures, ENCLOSURE_NAMES)
    loss_total = syntax_total = region_total = smoothness_total = 0.0
    category_loss_totals = {name: 0.0 for name in SAMPLE_KINDS}
    category_counts = {name: 0 for name in SAMPLE_KINDS}
    predicted_switches = false_switches = target_switches = adjacent_count = 0
    batch_index = 0
    for batch_index, batch in enumerate(loader, 1):
        stop_requested = False
        non_blocking = device.type == "cuda"
        inputs = batch["input_ids"].to(device, non_blocking=non_blocking)
        labels = batch["labels"].to(device, non_blocking=non_blocking)
        regions = batch["region_labels"].to(device, non_blocking=non_blocking)
        enclosures = batch["enclosure_labels"].to(device, non_blocking=non_blocking)
        language = batch["language_id"].to(device, non_blocking=non_blocking)
        loss_weights = batch["loss_weights"].to(device, non_blocking=non_blocking)
        sample_kinds = batch["sample_kind_id"].to(device, non_blocking=non_blocking)
        loss_weights[sample_kinds == SAMPLE_KINDS.index("long_string")] *= (
            config.long_string_loss_scale
        )
        if training and config.host_hint_dropout:
            language = language.masked_fill(
                torch.rand(language.shape, device=device) < config.host_hint_dropout, 0
            )
        state = model.initial_state(inputs.shape[0], device)
        chunk_losses = []
        syntax_losses = []
        region_losses = []
        enclosure_losses = []
        smoothness_losses = []
        category_losses: dict[str, list[Tensor]] = {name: [] for name in SAMPLE_KINDS}
        previous_region_logits = None
        previous_region_targets = None
        previous_region_predictions = None
        with torch.set_grad_enabled(training):
            for chunk_index in range(inputs.shape[1]):
                syntax_logits, region_logits, enclosure_logits, state = model.forward_chunk_with_aux(
                    inputs[:, chunk_index], state, config.chunk_length, language
                )
                if not torch.any(labels[:, chunk_index] != IGNORE_LABEL_ID):
                    continue
                syntax_values = criterion(
                    syntax_logits.reshape(-1, syntax_logits.shape[-1]),
                    labels[:, chunk_index].reshape(-1),
                )
                region_values = criterion(
                    region_logits.reshape(-1, region_logits.shape[-1]),
                    regions[:, chunk_index].reshape(-1),
                )
                enclosure_values = criterion(
                    enclosure_logits.reshape(-1, enclosure_logits.shape[-1]),
                    enclosures[:, chunk_index].reshape(-1),
                )
                valid = labels[:, chunk_index].reshape(-1) != IGNORE_LABEL_ID
                weights = loss_weights[:, chunk_index].reshape(-1)
                denominator = weights[valid].sum().clamp_min(1)
                syntax_loss = (syntax_values[valid] * weights[valid]).sum() / denominator
                region_loss = (region_values[valid] * weights[valid]).sum() / denominator
                enclosure_loss = (
                    enclosure_values[valid] * weights[valid]
                ).sum() / denominator
                syntax_by_sample = syntax_values.view(inputs.shape[0], -1)
                region_by_sample = region_values.view(inputs.shape[0], -1)
                valid_by_sample = valid.view(inputs.shape[0], -1)
                weights_by_sample = weights.view(inputs.shape[0], -1)
                for kind_id, kind_name in enumerate(SAMPLE_KINDS):
                    kind_mask = sample_kinds == kind_id
                    if kind_mask.any():
                        kind_valid = valid_by_sample[kind_mask]
                        kind_weights = weights_by_sample[kind_mask]
                        category_losses[kind_name].append(
                            (
                                syntax_by_sample[kind_mask][kind_valid]
                                + config.region_loss_weight
                                * region_by_sample[kind_mask][kind_valid]
                            ).mul(kind_weights[kind_valid]).sum() / denominator
                        )
                smoothness_loss = region_smoothness_loss(
                    region_logits, regions[:, chunk_index],
                    previous_region_logits, previous_region_targets,
                )
                syntax_losses.append(syntax_loss)
                region_losses.append(region_loss)
                enclosure_losses.append(enclosure_loss)
                smoothness_losses.append(smoothness_loss)
                chunk_losses.append(
                    syntax_loss
                    + config.region_loss_weight * region_loss
                    + config.enclosure_loss_weight * enclosure_loss
                    + config.region_smoothness_weight * smoothness_loss
                )
                syntax_metrics.update(syntax_logits, labels[:, chunk_index])
                region_metrics.update(region_logits, regions[:, chunk_index])
                enclosure_metrics.update(enclosure_logits, enclosures[:, chunk_index])
                predictions = region_logits.argmax(dim=-1)
                current_targets = regions[:, chunk_index]
                pair_predictions = predictions[:, 1:] != predictions[:, :-1]
                pair_targets = current_targets[:, 1:] != current_targets[:, :-1]
                pair_valid = (
                    (current_targets[:, 1:] != IGNORE_LABEL_ID)
                    & (current_targets[:, :-1] != IGNORE_LABEL_ID)
                )
                if previous_region_predictions is not None and previous_region_targets is not None:
                    pair_predictions = torch.cat((
                        (predictions[:, :1] != previous_region_predictions[:, -1:]),
                        pair_predictions,
                    ), dim=1)
                    pair_targets = torch.cat((
                        (current_targets[:, :1] != previous_region_targets[:, -1:]),
                        pair_targets,
                    ), dim=1)
                    pair_valid = torch.cat((
                        (
                            (current_targets[:, :1] != IGNORE_LABEL_ID)
                            & (previous_region_targets[:, -1:] != IGNORE_LABEL_ID)
                        ),
                        pair_valid,
                    ), dim=1)
                predicted_switches += int((pair_predictions & pair_valid).sum())
                target_switches += int((pair_targets & pair_valid).sum())
                false_switches += int((pair_predictions & ~pair_targets & pair_valid).sum())
                adjacent_count += int(pair_valid.sum())
                previous_region_logits = region_logits
                previous_region_targets = current_targets
                previous_region_predictions = predictions
            loss = torch.stack(chunk_losses).mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                category_gradient_norms = {}
                category_batch_values = {}
                for kind_name, values in category_losses.items():
                    if not values:
                        continue
                    category_loss = torch.stack(values).mean()
                    category_loss_totals[kind_name] += float(category_loss.detach())
                    category_counts[kind_name] += 1
                    category_batch_values[kind_name] = float(category_loss.detach())
                    if (
                        config.category_gradient_every_steps > 0
                        and batch_index % config.category_gradient_every_steps == 0
                    ):
                        gradient = torch.autograd.grad(
                            category_loss, model.forward_encoder.weight_hh_l0,
                            retain_graph=True, allow_unused=True,
                        )[0]
                        category_gradient_norms[kind_name] = (
                            0.0 if gradient is None else float(gradient.norm())
                        )
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
                stop_requested = bool(
                    after_optimizer_step is not None and after_optimizer_step()
                )
        syntax_mean = torch.stack(syntax_losses).mean()
        region_mean = torch.stack(region_losses).mean()
        smoothness_mean = torch.stack(smoothness_losses).mean()
        loss_total += float(loss.detach())
        syntax_total += float(syntax_mean.detach())
        region_total += float(region_mean.detach())
        smoothness_total += float(smoothness_mean.detach())
        if training and step_callback is not None and (
            batch_index % config.log_every_steps == 0
            or category_gradient_norms
            or batch_index == len(loader)
        ):
            step_callback(
                {
                    "batch": batch_index,
                    "loss": float(loss.detach()),
                    "gradient_norm": gradient_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **{
                        f"category/{name}/gradient_norm": value
                        for name, value in category_gradient_norms.items()
                    },
                    **{
                        f"category/{name}/loss_contribution": value
                        for name, value in category_batch_values.items()
                    },
                }
            )
        if training and stop_requested:
            break
    result = syntax_metrics.compute()
    batches = max(1, batch_index)
    result.update(
        {
            "loss": loss_total / batches,
            "syntax_loss": syntax_total / batches,
            "region_loss": region_total / batches,
            "region_smoothness_loss": smoothness_total / batches,
            "category_loss": {
                name: category_loss_totals[name] / max(1, category_counts[name])
                for name in SAMPLE_KINDS if category_counts[name]
            },
            "region": region_metrics.compute(),
            "enclosure": enclosure_metrics.compute(),
        }
    )
    result["region"].update({
        "predicted_switch_rate": predicted_switches / max(1, adjacent_count),
        "target_switch_rate": target_switches / max(1, adjacent_count),
        "false_switch_rate": false_switches / max(1, adjacent_count),
    })
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
        prose_code_fraction=config.prose_code_fraction,
        mixed_language_fraction=config.mixed_language_fraction,
        string_code_fraction=config.string_code_fraction,
        long_string_fraction=config.long_string_fraction,
        mixed_ordinary_fraction=config.mixed_ordinary_fraction,
        embedded_loss_weight=config.embedded_loss_weight,
        quota_cycle=config.batch_size,
    )
    validation_data = StreamingFragmentDataset(
        validation_paths, config.chunk_length, config.lookahead, config.chunks_per_sample,
        config.validation_samples, config.seed + 1,
        delimiter_wrap_fraction=0.0,
        prose_code_fraction=0.0,
    )
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    # Augmentation kinds are distributed over stable 16-sample cycles, so an
    # unshuffled loader gives every batch a predictable mix while source crops
    # still change deterministically each epoch.
    train_loader = DataLoader(train_data, shuffle=False, **loader_kwargs)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_kwargs)
    model = StreamingByteGRU(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"training on {device} | {model.parameter_count:,} parameters | "
        f"batch={config.batch_size} chunks={config.chunks_per_sample} "
        f"chunk_length={config.chunk_length}"
    )
    run = None
    if wandb_project:
        import wandb

        run = wandb.init(
            project=wandb_project, name=wandb_name, job_type="train-streaming",
            dir=str(output), config={**asdict(config), **model_config.to_dict()},
        )
    stopper = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)
    ema = ExponentialMovingAverage(model, config.ema_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=config.lr_decay_factor,
        patience=config.lr_plateau_patience, min_lr=config.minimum_learning_rate,
    )
    best_path = output / "best.pt"
    specialized_best = {
        "ordinary": (-float("inf"), output / "best-ordinary.pt"),
        "embedded": (-float("inf"), output / "best-embedded.pt"),
        "string-embedded": (-float("inf"), output / "best-string-embedded.pt"),
        "long-string": (-float("inf"), output / "best-long-string.pt"),
    }
    global_step = 0
    current_epoch = 0
    stop_requested = False
    selection_history: deque[float] = deque(maxlen=config.selection_smoothing_window)
    best_candidate: tuple[float, dict[str, object]] | None = None
    best_eligible_score = -float("inf")

    def log_step(values):
        payload = {"global_step": global_step, **{f"train_step/{k}": v for k, v in values.items()}}
        print(
            f"step {global_step:>6} train | loss {values['loss']:.4f} | "
            f"grad {values['gradient_norm']:.3f} | lr {values['learning_rate']:.2e}"
        )
        category_values = {
            key.removeprefix("category/"): value
            for key, value in values.items() if key.startswith("category/")
        }
        if any(key.endswith("gradient_norm") for key in category_values):
            gradients = " ".join(
                f"{key.removesuffix('/gradient_norm')}={value:.3f}"
                for key, value in category_values.items()
                if key.endswith("gradient_norm")
            )
            print(f"             category gradients | {gradients}")
        if run:
            run.log(payload, step=global_step)

    def evaluate_full() -> bool:
        """Run authoritative validation and update deployment checkpoints."""
        nonlocal best_candidate, best_eligible_score, stop_requested
        with ema.apply_to(model), torch.no_grad():
            validation_metrics = run_stream_epoch(model, validation_loader, device, config)
            challenge_metrics = evaluate_challenge(
                model, device, config.chunk_length, config.lookahead
            )
        model.train(True)
        ordinary_score = float(validation_metrics["macro_f1"])
        challenge_score = float(challenge_metrics["score"])
        raw_selection = (
            2 * ordinary_score * challenge_score
            / max(ordinary_score + challenge_score, 1e-12)
        )
        selection_history.append(raw_selection)
        selection_score = sum(selection_history) / len(selection_history)
        scheduler.step(selection_score)
        floors = {
            "ordinary": ordinary_score >= config.minimum_ordinary_score,
            "embedded": float(challenge_metrics["embedded_code"]["macro_f1"])
            >= config.minimum_embedded_score,
            "string_embedded": float(
                challenge_metrics["embedded_string_code"]["macro_f1"]
            ) >= config.minimum_string_embedded_score,
            "long_string": float(challenge_metrics["long_string"]["accuracy"])
            >= config.minimum_long_string_accuracy,
        }
        eligible = all(floors.values())
        _, plateau_stop, _ = stopper.update(selection_score)
        checkpoint: dict[str, object] = {
            "architecture": "streaming-gru",
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "language_schema_version": LANGUAGE_SCHEMA_VERSION,
            "model_config": model.config.to_dict(),
            "train_config": asdict(config),
            # Deployment and ONNX conversion consume the stable EMA weights.
            "model_state": ema.state_dict(),
            "training_model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "ema_decay": config.ema_decay,
            "ema_updates": ema.updates,
            "epoch": current_epoch,
            "global_step": global_step,
            "metrics": validation_metrics,
            "challenge_metrics": challenge_metrics,
            "raw_selection_score": raw_selection,
            "selection_score": selection_score,
            "selection_history": list(selection_history),
            "skill_floors": floors,
        }
        torch.save(checkpoint, output / "last.pt")
        if best_candidate is None or raw_selection > best_candidate[0]:
            best_candidate = (raw_selection, checkpoint)
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, output / "best-observed.pt")
        if eligible and selection_score > best_eligible_score:
            torch.save(checkpoint, output / "best-qualified.pt")
            best_eligible_score = selection_score
        specialized_values = {
            "ordinary": ordinary_score,
            "embedded": float(challenge_metrics["embedded_code"]["macro_f1"]),
            "string-embedded": float(
                challenge_metrics["embedded_string_code"]["macro_f1"]
            ),
            "long-string": float(challenge_metrics["long_string"]["accuracy"]),
        }
        for name, value in specialized_values.items():
            previous, path = specialized_best[name]
            if value > previous:
                torch.save(checkpoint, path)
                specialized_best[name] = (value, path)
        print(
            f"step {global_step:>6} valid | f1 {ordinary_score:.4f} | "
            f"challenge {challenge_score:.4f} | select {raw_selection:.4f} "
            f"(smooth {selection_score:.4f}) | lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        print(
            "             skills | "
            f"ordinary {challenge_metrics['ordinary_code']['macro_f1']:.3f} | "
            f"comments {challenge_metrics['embedded_code']['macro_f1']:.3f} | "
            f"strings {challenge_metrics['embedded_string_code']['macro_f1']:.3f} | "
            f"prose {challenge_metrics['prose']['accuracy']:.3f} | "
            f"long {challenge_metrics['long_string']['accuracy']:.3f} | "
            f"floors {'yes' if eligible else 'no'}"
        )
        if run:
            run.log({
                "epoch": current_epoch,
                "validation/loss": validation_metrics["loss"],
                "validation/accuracy": validation_metrics["accuracy"],
                "validation/macro_f1": validation_metrics["macro_f1"],
                "challenge/score": challenge_metrics["score"],
                "selection/raw_score": raw_selection,
                "selection/smoothed_score": selection_score,
                "selection/floors_met": int(eligible),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                **{
                    f"challenge/{name}/{metric}": values[metric]
                    for name, values in challenge_metrics.items()
                    if isinstance(values, dict)
                    for metric in ("accuracy", "macro_f1")
                    if metric in values
                },
            }, step=global_step)
        stop_requested = bool(
            plateau_stop and eligible and global_step >= config.minimum_training_steps
        )
        return stop_requested

    def after_optimizer_step() -> bool:
        nonlocal global_step
        global_step += 1
        ema.update(model)
        if global_step % config.validation_every_steps == 0:
            return evaluate_full()
        if config.probe_every_steps and global_step % config.probe_every_steps == 0:
            with ema.apply_to(model), torch.no_grad():
                model.eval()
                probe = evaluate_challenge(
                    model, device, config.chunk_length, config.lookahead
                )
            model.train(True)
            print(
                f"step {global_step:>6} probe | score {probe['score']:.4f} | "
                f"ordinary {probe['ordinary_code']['macro_f1']:.3f} | "
                f"comments {probe['embedded_code']['macro_f1']:.3f} | "
                f"strings {probe['embedded_string_code']['macro_f1']:.3f} | "
                f"prose {probe['prose']['accuracy']:.3f} | "
                f"long {probe['long_string']['accuracy']:.3f}"
            )
            if run:
                run.log({
                    "probe/score": probe["score"],
                    **{
                        f"probe/{name}/{metric}": values[metric]
                        for name, values in probe.items() if isinstance(values, dict)
                        for metric in ("accuracy", "macro_f1") if metric in values
                    },
                }, step=global_step)
        return False

    try:
        epochs = itertools.count(1) if config.epochs <= 0 else range(1, config.epochs + 1)
        for epoch in epochs:
            current_epoch = epoch
            train_data.set_epoch(epoch)
            run_stream_epoch(
                model, train_loader, device, config, optimizer, log_step,
                after_optimizer_step,
            )
            if stop_requested:
                break
        if global_step % config.validation_every_steps != 0:
            evaluate_full()
        if run:
            run.log_model(str(best_path), name=f"{run.id}-streaming-gru")
    finally:
        if run:
            run.finish()
    onnx_path = export_streaming_checkpoint(best_path, output / "best.onnx")
    print(f"exported ONNX: {onnx_path}")
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
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--delimiter-wrap-fraction", type=float, default=0.1)
    parser.add_argument("--prose-code-fraction", type=float, default=0.15)
    parser.add_argument("--mixed-language-fraction", type=float, default=0.0)
    parser.add_argument("--string-code-fraction", type=float, default=0.0)
    parser.add_argument("--long-string-fraction", type=float, default=0.0)
    parser.add_argument("--mixed-ordinary-fraction", type=float, default=0.0)
    parser.add_argument("--embedded-loss-weight", type=float, default=2.0)
    parser.add_argument("--region-smoothness-weight", type=float, default=0.03)
    parser.add_argument("--probe-every-steps", type=int, default=250)
    parser.add_argument("--validation-every-steps", type=int, default=1000)
    parser.add_argument("--minimum-training-steps", type=int, default=20000)
    parser.add_argument("--selection-smoothing-window", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--lr-plateau-patience", type=int, default=4)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--minimum-ordinary-score", type=float, default=0.90)
    parser.add_argument("--minimum-embedded-score", type=float, default=0.75)
    parser.add_argument("--minimum-string-embedded-score", type=float, default=0.75)
    parser.add_argument("--minimum-long-string-accuracy", type=float, default=0.95)
    parser.add_argument("--long-string-loss-scale", type=float, default=0.5)
    parser.add_argument("--category-gradient-every-steps", type=int, default=250)
    parser.add_argument("--enclosure-loss-weight", type=float, default=0.1)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("--byte-embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token-context-dim", type=int, default=0)
    parser.add_argument("--token-context-radius", type=int, default=2)
    parser.add_argument("--region-conditioned-syntax", action="store_true")
    parser.add_argument(
        "--allow-syntax-gradient-into-region", action="store_true",
        help="allow syntax loss to shape region probabilities used by syntax adapters",
    )
    args = parser.parse_args()
    config = StreamingTrainConfig(
        chunk_length=args.chunk_length, lookahead=args.lookahead,
        chunks_per_sample=args.chunks_per_sample, batch_size=args.batch_size,
        train_samples=args.train_samples, validation_samples=args.validation_samples,
        epochs=args.epochs, early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate, num_workers=args.num_workers,
        log_every_steps=args.log_every_steps, device=args.device,
        delimiter_wrap_fraction=args.delimiter_wrap_fraction,
        prose_code_fraction=args.prose_code_fraction,
        mixed_language_fraction=args.mixed_language_fraction,
        string_code_fraction=args.string_code_fraction,
        long_string_fraction=args.long_string_fraction,
        mixed_ordinary_fraction=args.mixed_ordinary_fraction,
        embedded_loss_weight=args.embedded_loss_weight,
        region_smoothness_weight=args.region_smoothness_weight,
        probe_every_steps=args.probe_every_steps,
        validation_every_steps=args.validation_every_steps,
        minimum_training_steps=args.minimum_training_steps,
        selection_smoothing_window=args.selection_smoothing_window,
        ema_decay=args.ema_decay,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_decay_factor=args.lr_decay_factor,
        minimum_learning_rate=args.minimum_learning_rate,
        minimum_ordinary_score=args.minimum_ordinary_score,
        minimum_embedded_score=args.minimum_embedded_score,
        minimum_string_embedded_score=args.minimum_string_embedded_score,
        minimum_long_string_accuracy=args.minimum_long_string_accuracy,
        long_string_loss_scale=args.long_string_loss_scale,
        category_gradient_every_steps=args.category_gradient_every_steps,
        enclosure_loss_weight=args.enclosure_loss_weight,
    )
    best_path = train_streaming(
        args.train, args.validation, args.output, config,
        StreamingGRUConfig(
            byte_embedding_dim=args.byte_embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            region_conditioned_syntax=args.region_conditioned_syntax,
            detach_region_for_syntax=not args.allow_syntax_gradient_into_region,
            token_context_dim=args.token_context_dim,
            token_context_radius=args.token_context_radius,
        ),
        args.wandb_project, args.wandb_name,
    )
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
