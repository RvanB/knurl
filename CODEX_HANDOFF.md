# Codex Development Handoff: Neural Highlight

This document is the working context for continuing development on the CUDA
machine. Read it together with `README.md` and
`neural-syntax-highlighter-plan.md`, but treat this file as the record of the
latest product decisions and known problems.

## Mission

Build a small, fast syntax highlighter that consumes raw UTF-8 bytes and emits
one visual syntax class per byte. Tree-sitter is a training-data teacher only;
inference must not parse the input. The eventual target is interactive editor
highlighting, including arbitrary note buffers containing abrupt and possibly
unrealistic changes between programming languages.

The user explicitly values visual usefulness over determining whether code is
executable. Code displayed inside comment delimiters should still receive code
syntax colors. Natural-language prose should receive the `comment` color.

## Current product decisions

1. Input is raw bytes, not abstract token shapes. Keyword spelling and other
   lexical information are important.
2. The model predicts both a syntax class and a local language region per byte.
3. Mixed-language training examples do not need to form valid files. Arbitrary
   concatenation is desirable because an editor buffer may be a collection of
   notes and code snippets.
4. Stateful inference is required. There is no finite context window large
   enough to resolve every left-context construct, so the streaming model
   carries forward state and uses bounded right lookahead.
5. Comment delimiters are not an instruction to color their entire contents
   gray. Prose is `comment`; recognizable embedded code retains normal syntax
   labels.
6. Real training comments are sanitized during dataset preparation. Fenced
   blocks and conservatively code-like lines are removed, and all remaining
   prose is fully supervised as `comment`. There should be no cancelled loss
   for retained comment bodies.
7. Embedded code in prose/comments is created synthetically so its labels are
   known exactly. Some synthetic mixtures have delimiters and some do not.
8. Do not add MPS-specific work. Development may happen on a Mac, but serious
   training runs on an Nvidia CUDA machine with 12 GB VRAM.

## Label and language vocabulary

The syntax classes are defined in `neural_highlight/labels.py`:

`plain`, `keyword`, `identifier`, `function`, `function_definition`,
`parameter`, `type`, `property`, `string`, `comment`, `number`, `operator`,
`punctuation`, and `constant`.

The teacher supports Python, JavaScript, TypeScript, HTML, CSS, Rust, C, C++,
Go, and Java. Language IDs are in `neural_highlight/languages.py`.

## Architecture and important files

- `neural_highlight/dataset/annotate.py`: Tree-sitter teacher, comment
  sanitization, comment supervision, and normalized per-byte labels.
- `neural_highlight/dataset/prepare.py`: streams Stack records, sanitizes each
  source file, annotates the whole file, then repository-splits it.
- `neural_highlight/dataset/storage.py`: lazy indexed/mmap JSONL storage. Do not
  regress to loading all decoded records into RAM.
- `neural_highlight/dataset/prose.py`: conservative prose extraction used for
  synthetic prose/code samples.
- `neural_highlight/dataset/fragments.py`: stateless crops, multilingual random
  mixtures, delimiter wrapping, and prose/code augmentation.
- `neural_highlight/dataset/streams.py`: consecutive training chunks for the
  persistent-state model, plus delimiter and prose/code augmentation.
- `neural_highlight/models/bigru.py`: original stateless bidirectional GRU.
- `neural_highlight/models/streaming_gru.py`: forward GRU state across committed
  chunks plus a reverse GRU over bounded lookahead.
- `neural_highlight/train.py`: mature stateless trainer with detailed W&B
  metrics and DataLoader tuning.
- `neural_highlight/train_streaming.py`: current stateful trainer. This is the
  relevant path for future work, but its evaluation and W&B support need work.
- `neural_highlight/stream_infer.py`: arbitrarily long stateful inference and
  ANSI/debug span output.

The streaming default model is intentionally tiny: 32-dimensional byte
embeddings, hidden size 64, two GRU layers, and an 8-dimensional language
embedding. It has separate syntax and language-region heads.

## Set up the CUDA machine

From the repository root:

```bash
uv sync
uv run wandb login
```

The Stack Smol dataset is gated. Accept its Hugging Face terms and authenticate
if the cache is not already populated:

```bash
uv run hf auth login
```

Keep `.hf-cache`; it avoids downloading Stack data again. Dataset JSONL files
are safe to overwrite.

## Regenerate the current dataset

The latest sanitized-comment policy is a preparation-time transformation, so
datasets made under the older masking policy must be regenerated after pulling
the latest code:

```bash
HF_HOME=.hf-cache UV_CACHE_DIR=.uv-cache \
uv run python scripts/prepare_multilingual.py \
  --source smol \
  --max-files 5000 \
  --max-bytes 1000000
```

Expected paths are:

```text
data/annotated/{python,javascript,typescript,html,css,rust,c,c++,go,java}/
  train.jsonl
  validation.jsonl
  test.jsonl
```

The current writer may still serialize `label_mask_b64`, but regenerated
records should have fully enabled masks for retained comments. The important
invariant is that code-like comment content has been removed and retained
comment prose has `comment` labels.

## Current streaming training command

Use a new output directory after changing the data policy; do not compare a
continued old checkpoint as though it used the new objective.

```bash
uv run python scripts/train_streaming_model.py \
  --train data/annotated/{python,javascript,typescript,html,css,rust,c,c++,go,java}/train.jsonl \
  --validation data/annotated/{python,javascript,typescript,html,css,rust,c,c++,go,java}/validation.jsonl \
  --output runs/multilingual-streaming-prose \
  --device cuda \
  --batch-size 16 \
  --num-workers 2 \
  --chunk-length 256 \
  --lookahead 128 \
  --chunks-per-sample 8 \
  --train-samples 4096 \
  --validation-samples 512 \
  --prose-code-fraction 0.15 \
  --delimiter-wrap-fraction 0.1 \
  --log-every-steps 20 \
  --early-stopping-patience 3 \
  --wandb-project neural-highlight \
  --wandb-name multilingual-streaming-prose
```

There is no epoch cap when `--epochs` is zero, which is the default. Training
stops after validation accuracy fails to improve for three epochs. Start with
batch size 16 on the 12 GB GPU, then measure before increasing it.

## Inference

Colored output:

```bash
uv run python -m neural_highlight.stream_infer \
  runs/multilingual-streaming-prose/best.pt \
  path/to/sample.txt \
  --language unknown \
  --device cuda \
  --color
```

Omit `--color` for byte spans, predicted syntax classes, and predicted language
regions. The inference CLI reads chunk length and lookahead from the checkpoint.
For an eventual editor integration, benchmark CPU inference as well; CUDA
startup/transfer overhead may lose for small edits.

## Current quality problem

Inference from the first streaming experiments was poor. Do not assume that
more epochs with the same metric will solve it. The earlier comment-masking
policy also made prose inherently unlearnable; the sanitizer/full-supervision
change addresses that data defect, but it has not yet been demonstrated by a
good run.

Important remaining weaknesses:

1. Streaming early stopping uses overall byte accuracy. Frequent classes can
   dominate it, and it does not directly measure prose/code behavior.
2. Synthetic prose/code augmentation is disabled for validation. Real sanitized
   prose is now validated, but embedded-code behavior still has no dedicated,
   deterministic validation suite.
3. `StreamingFragmentDataset` currently samples one source record/language for
   a sequence. Unlike `FragmentDataset`, it does not yet synthesize arbitrary
   transitions between multiple languages in one stream. This conflicts with
   a core product goal.
4. Synthetic streaming prose/code samples currently have a simple prose-then-
   code structure. They should include multiple alternating spans, inline code,
   several languages, and boundary crossings between committed chunks.
5. The model is very small and its size is not exposed by the streaming CLI.
6. The streaming trainer logs much less detail to W&B than the stateless
   trainer. It lacks per-class metrics in the dashboard, throughput/GPU memory,
   representative colored examples, and richer checkpoint handling.
7. Training defaults to only 4096 sampled sequences per epoch across ten
   languages. That is likely insufficient for a serious experiment.
8. Identifier coloring can vary between repeated occurrences. A non-ML
   consistency decoder was discussed, but the user explicitly chose to defer
   it until the prose/code training problem is fixed.

## Recommended next work

Prioritize measurement before another long training run:

1. Build a deterministic held-out challenge set containing sanitized prose,
   code inside delimiters, delimiter-free prose/code notes, multiple alternating
   spans, long block boundaries, and abrupt language changes.
2. Report per-class precision/recall/F1, especially `comment`, plus separate
   metrics for prose bytes, embedded-code bytes, language transitions, and
   ordinary source.
3. Add a balanced early-stopping metric (likely macro F1 or a named composite)
   to the streaming trainer and save best/after-best/last checkpoints.
4. Add actual multi-language stream synthesis to `StreamingFragmentDataset`.
5. Log a fixed panel of colored qualitative examples to W&B each epoch.
6. Only then compare larger models, more samples per epoch, augmentation
   fractions, and loss weighting. Expose embedding size, hidden size, layer
   count, and perhaps dropout as CLI flags.

Do not optimize for ONNX yet. Once quality is credible, benchmark PyTorch CPU,
then export a chunk-level `state_in -> predictions + state_out` contract to
ONNX and measure editor-scale latency.

## Verification and repository hygiene

Run before committing:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest -q
UV_CACHE_DIR=.uv-cache uv run ruff check neural_highlight tests
git diff --check
```

At the time this handoff was created, the suite had 40 passing tests and lint
was clean. Preserve user data and unrelated worktree changes. Do not commit
`.DS_Store`, dataset caches, generated annotations, run directories, W&B data,
or checkpoints.

## Suggested first message to a new Codex agent

```text
Read CODEX_HANDOFF.md, README.md, and neural-syntax-highlighter-plan.md in full.
Inspect the current git status and recent commits without discarding any work.
The immediate problem is poor inference quality in the stateful multilingual
highlighter. First implement a deterministic prose/code and mixed-language
challenge validation set with useful per-slice metrics and W&B reporting. Do
not begin another large training run until the validation metric can measure
the behaviors described in the handoff. Use uv for all Python commands and
target CUDA, not MPS.
```
