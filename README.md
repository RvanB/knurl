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

Preparation removes fenced blocks and conservatively code-like lines from
Tree-sitter comment captures. Original bodies are replaced with diverse,
deterministic generated prose while delimiters and layout are retained. The
result is fully supervised as `comment`; comment losses are not masked. Another 10% of samples
wrap genuine annotated code in comment delimiters while preserving its code
labels, preventing `/*` or `<!--` from becoming a blanket "everything is a
comment" cue. Configure this with `--delimiter-wrap-fraction`.

Older annotation files are rejected because they do not have this generated-comment
policy. Regenerate annotations before training a new model.

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

Render a file using the model's predicted syntax classes:

```sh
uv run python -m neural_highlight.infer runs/multilingual-bigru/best.pt \
  notes.txt --language unknown --device cuda --color
```

Inline text and stdin are also supported:

```sh
uv run python -m neural_highlight.infer runs/multilingual-bigru/best.pt \
  --text 'const answer = 42;' --language unknown --device cuda --color

printf 'def hello():\n    return 42\n' | uv run python -m neural_highlight.infer \
  runs/multilingual-bigru/best.pt - --language unknown --device cuda --color
```

Omit `--color` to print byte ranges with both predicted local language and
syntax class, which is useful for debugging classification behavior.

## Train the persistent-state model

The streaming model carries a forward GRU state across committed 256-byte
chunks and uses 128 bytes of bounded right lookahead. Training backpropagates
through eight consecutive chunks by default:

```sh
uv run python scripts/train_streaming_model.py \
  --train data/annotated/python/train.jsonl data/annotated/rust/train.jsonl \
  --validation data/annotated/python/validation.jsonl data/annotated/rust/validation.jsonl \
  --output runs/streaming-gru --device cuda \
  --chunk-length 256 --lookahead 128 --chunks-per-sample 8 \
  --batch-size 16 --epochs 0 --early-stopping-patience 30 \
  --wandb-project neural-highlight --wandb-name streaming-gru
```

At the end of training, `best.pt` is automatically exported and numerically
verified as `best.onnx`. The deployment contract explicitly accepts `state_in`
and returns `state_out`, making it suitable for editor-side state checkpoints.
Export an existing compatible checkpoint independently with:

```sh
uv run python scripts/export_streaming_onnx.py runs/streaming-gru/best.pt
```

Run arbitrarily long stateful inference with checkpoint-derived chunk settings:

```sh
uv run python -m neural_highlight.stream_infer runs/streaming-gru/best.onnx \
  long-comment.rs --language rust --device cuda --color
```

Streaming inference uses ONNX Runtime. Chunk length, lookahead, model shape,
and vocabulary schema are embedded in the ONNX metadata and validated on load.

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
Go, Java, Shell/Bash, SQL, Markdown, Lua, Ruby, and GLSL. Prepare all sixteen
monolingual corpora with one command:

```sh
uv run python scripts/prepare_multilingual.py --source smol --max-files 5000
```

GLSL automatically falls back to the public Smol-XS split because it is not in
the 30-language Smol corpus. Sampling is balanced by language rather than raw
file count.

Then pass all desired splits to the trainer. The fragment sampler draws across
the combined record pool and synthesizes abrupt language changes at runtime:

```sh
uv run python scripts/train_model.py \
  --train data/annotated/{python,javascript,typescript,html,css,rust,c,c++,go,java,shell,sql,markdown,lua,ruby,glsl}/train.jsonl \
  --validation data/annotated/{python,javascript,typescript,html,css,rust,c,c++,go,java,shell,sql,markdown,lua,ruby,glsl}/validation.jsonl \
  --output runs/multilingual-bigru --device cuda --language-embedding \
  --mixture-fraction 0.25 --max-regions 4 --wandb
```

During training, `--prose-code-fraction` (default `0.15`) extracts prose from
the generated, fully supervised comments and mixes it with genuine labeled
code. The prose receives the `comment` target while the code retains its syntax
targets; half of these synthetic examples omit comment delimiters. Thus all
embedded-code examples have labels known by construction. Validation does not
use this augmentation. Set the fraction to zero to disable it.

Streaming training can reserve `--long-string-fraction` for multiline literals
whose entire contents remain `string`. With an unshuffled loader and a quota
cycle equal to the batch size, augmentation fractions become per-batch replay
quotas rather than merely epoch-wide probabilities. This keeps long ordinary
strings and syntax-highlighted embedded-code strings in the same optimizer
updates.

Streaming evaluation is step-based. `--probe-every-steps` runs the fixed
behavioral challenge without affecting checkpoint decisions, while
`--validation-every-steps` runs the authoritative frozen validation crops and
challenge. Deployment checkpoints use EMA weights. Selection is smoothed over
`--selection-smoothing-window` checks, enforces minimum ordinary/embedded/long-
string skill floors, decays the learning rate on a plateau, and honors
`--minimum-training-steps` before early stopping.

For diagnosing replay specialization, each sample carries its augmentation
category into the trainer. W&B logs each category's contribution to the shared
batch loss and periodically measures its gradient norm at the recurrent
encoder. `--long-string-loss-scale` caps the effective long-string contribution
without changing its targets or byte coverage.

V17 separates rendered syntax from structural enclosure. Comment and string
bytes use `plain` as their content target, while a training-only auxiliary head
predicts `code`, `string`, or `comment`. Embedded code retains its underlying
syntax labels. The auxiliary loss defaults to `0.1` and is not exported as an
ONNX output.

V18 adds language-agnostic lexical-run context to the syntax head. Alphabetic,
underscore, and non-ASCII byte runs are pooled, combined with two neighboring
runs on either side, projected to 32 features, and broadcast back to their
bytes. This gives prose/code decisions word-pattern context without changing
the byte-level public output or requiring a Tree-sitter tokenizer at inference.

For editor integrations, keep one ONNX session alive with the JSON-lines
server:

```sh
uv run python -m neural_highlight.stream_server \
  runs/multilingual-streaming-comments-v18/best.onnx --device cpu
```

Write one JSON object per stdin line:

```json
{"id":1,"text":"def hello():\n    return 42","language":"unknown"}
```

The server flushes one response per request. Each span contains `start` and
`end` UTF-8 byte offsets, named `syntax` and `language` classes, and their
numeric `syntax_id` and `language_id`. The request `id` is echoed unchanged so
clients can correlate responses. Use `text_base64` instead of `text` for
arbitrary bytes, and `--describe` to print all ID mappings to stderr. Stdout
remains machine-readable JSON only.
