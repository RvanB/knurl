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

By default, another 25% of samples are deliberately unrealistic mixtures of
two to four independently annotated fragments. Every byte carries both a syntax
target and a local language-region target. Mixed samples have an `unknown` host
hint, allowing a buffer to switch languages without being a valid container
document. Tune this with `--mixture-fraction` and `--max-regions`.

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

The model has separate syntax and per-byte language-region heads over its shared
encoder. Region loss defaults to 20% of the total loss, and host-language hints
are dropped for half of training samples. Configure these with
`--region-loss-weight` and `--host-hint-dropout`.

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

Training defaults to no epoch cap (`--epochs 0`) and stops when validation
accuracy has failed to improve for three epochs. Configure this with
`--early-stopping-patience`, `--early-stopping-min-delta`, and
`--early-stopping-metric {accuracy,macro_f1}`. `best.pt` is the latest improving
checkpoint, `after-best.pt` is the first epoch after that improvement, and
`last.pt` is the final completed epoch. A positive `--epochs` remains available
as a safety cap.

CUDA training defaults to four DataLoader workers, four prefetched batches per
worker, pinned host memory, persistent workers, and non-blocking device copies.
Tune the input pipeline with `--num-workers` and `--prefetch-factor`; use
`--no-pin-memory` or `--no-persistent-workers` only for troubleshooting. The
dataset epoch counter is shared so persistent workers still generate new crops
each epoch.

Optimizer-step telemetry is emitted every 50 batches by default. It includes
instantaneous/rolling loss, learning rate, gradient norm, throughput, and CUDA
memory. Change the cadence with `--log-every-steps`; validation and full
class-aware metrics remain epoch-level.

## Multilingual training

The teacher supports Python, JavaScript, TypeScript, HTML, CSS, Rust, C, C++,
Go, and Java. Prepare all ten monolingual corpora with one command:

```sh
uv run python scripts/prepare_multilingual.py --source smol --max-files 5000
```

Then pass all desired splits to the trainer. The fragment sampler draws across
the combined record pool and synthesizes abrupt language changes at runtime:

```sh
uv run python scripts/train_model.py \
  --train data/annotated/{python,javascript,typescript,html,css,rust,c,c++,go,java}/train.jsonl \
  --validation data/annotated/{python,javascript,typescript,html,css,rust,c,c++,go,java}/validation.jsonl \
  --output runs/multilingual-bigru --device cuda --language-embedding \
  --mixture-fraction 0.25 --max-regions 4 --wandb
```
