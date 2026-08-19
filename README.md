# Neural Highlight

An early experiment in predicting syntax highlighting directly from UTF-8 bytes.
The current milestone uses Tree-sitter as a teacher to produce one normalized
syntax label per byte of Python source.

```sh
uv sync
uv run python -m neural_highlight.dataset.annotate example.py
uv run pytest
```

Use `--spans` for a machine-readable-ish span listing, or omit it for an ANSI
colored preview plus a capture legend.

## Prepare a dataset

The Stack Smol is gated on Hugging Face. Accept its access conditions and log in
with `hf auth login`, then stream and annotate a bounded Python sample:

```sh
uv run python scripts/prepare_dataset.py --language python --max-files 1000
```

The command annotates each complete file before writing it to `train.jsonl`,
`validation.jsonl`, or `test.jsonl`. Splits are stable and repository-scoped;
source and byte labels are base64 encoded to keep each record on one JSON line.

For an unauthenticated bootstrap run, use `--source smol-xs`. The public XS
variant omits repository/path metadata, so its manifest records that splits use
stable content hashes instead of repositories; use regular Smol for real model
evaluation to prevent cross-repository leakage.

## Sample training fragments

```sh
uv run python scripts/show_fragments.py \
  data/annotated/python-sample/train.jsonl --length 256 --count 3
```

`FragmentDataset` emits padded PyTorch tensors and deterministically changes its
crops when `set_epoch()` is called. By default, 30% of crops are centered near a
non-plain syntax byte to reduce class imbalance.

## Train the first BiGRU

```sh
uv run python scripts/train_model.py \
  --train data/annotated/python-sample/train.jsonl \
  --validation data/annotated/python-sample/validation.jsonl \
  --output runs/python-bigru
```

Checkpoints include model/training configuration, optimizer state, epoch, and
validation metrics. Inference takes only a checkpoint, raw source bytes, and an
optional language ID; it does not invoke Tree-sitter.

### Watch training with Weights & Biases

Authenticate once with `uv run wandb login`, then add tracking flags:

```sh
uv run python scripts/train_model.py \
  --train data/annotated/python-sample/train.jsonl \
  --validation data/annotated/python-sample/validation.jsonl \
  --output runs/python-bigru \
  --wandb --wandb-project neural-highlight \
  --wandb-name python-bigru-256 \
  --wandb-tags python,bigru,context-256 \
  --log-every-steps 50
```

W&B receives hyperparameters, parameter count, losses, accuracy, macro F1,
per-class precision/recall/F1/support, gradient statistics, and the best model
checkpoint. Use `--wandb-mode offline` to test logging without an account, or
`--no-wandb-model` to keep checkpoints local.

Optimizer-step telemetry is emitted every 50 batches by default. It includes
instantaneous/rolling loss, learning rate, gradient norm, throughput, and CUDA
memory. Change the cadence with `--log-every-steps`; validation and full
class-aware metrics remain epoch-level.
