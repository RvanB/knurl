#!/usr/bin/env bash
set -euo pipefail

HF_HOME=.hf-cache UV_CACHE_DIR=.uv-cache \
  uv run python scripts/train_streaming_model.py \
    --train \
      data/annotated/python/train.jsonl \
      data/annotated/javascript/train.jsonl \
      data/annotated/typescript/train.jsonl \
      data/annotated/html/train.jsonl \
      data/annotated/css/train.jsonl \
      data/annotated/rust/train.jsonl \
      data/annotated/c/train.jsonl \
      data/annotated/c++/train.jsonl \
      data/annotated/go/train.jsonl \
      data/annotated/java/train.jsonl \
      data/annotated/shell/train.jsonl \
      data/annotated/sql/train.jsonl \
      data/annotated/markdown/train.jsonl \
      data/annotated/lua/train.jsonl \
      data/annotated/ruby/train.jsonl \
      data/annotated/glsl/train.jsonl \
    --validation \
      data/annotated/python/validation.jsonl \
      data/annotated/javascript/validation.jsonl \
      data/annotated/typescript/validation.jsonl \
      data/annotated/html/validation.jsonl \
      data/annotated/css/validation.jsonl \
      data/annotated/rust/validation.jsonl \
      data/annotated/c/validation.jsonl \
      data/annotated/c++/validation.jsonl \
      data/annotated/go/validation.jsonl \
      data/annotated/java/validation.jsonl \
      data/annotated/shell/validation.jsonl \
      data/annotated/sql/validation.jsonl \
      data/annotated/markdown/validation.jsonl \
      data/annotated/lua/validation.jsonl \
      data/annotated/ruby/validation.jsonl \
      data/annotated/glsl/validation.jsonl \
    --output runs/multilingual-streaming-comments-v18 \
    --device cuda \
    --batch-size 16 \
    --num-workers 2 \
    --chunk-length 256 \
    --lookahead 128 \
    --chunks-per-sample 8 \
    --train-samples 24576 \
    --validation-samples 1536 \
    --byte-embedding-dim 64 \
    --hidden-size 128 \
    --num-layers 2 \
    --dropout 0.1 \
    --region-conditioned-syntax \
    --token-context-dim 32 \
    --token-context-radius 2 \
    --prose-code-fraction 0.125 \
    --delimiter-wrap-fraction 0.125 \
    --mixed-language-fraction 0.125 \
    --string-code-fraction 0.125 \
    --long-string-fraction 0.0625 \
    --mixed-ordinary-fraction 0.125 \
    --embedded-loss-weight 2.0 \
    --region-smoothness-weight 0.03 \
    --log-every-steps 20 \
    --probe-every-steps 250 \
    --validation-every-steps 1000 \
    --minimum-training-steps 20000 \
    --selection-smoothing-window 5 \
    --ema-decay 0.999 \
    --lr-plateau-patience 4 \
    --lr-decay-factor 0.5 \
    --minimum-learning-rate 0.00001 \
    --minimum-ordinary-score 0.90 \
    --minimum-embedded-score 0.75 \
    --minimum-string-embedded-score 0.75 \
    --minimum-long-string-accuracy 0.95 \
    --long-string-loss-scale 0.5 \
    --category-gradient-every-steps 250 \
    --enclosure-loss-weight 0.1 \
    --early-stopping-patience 12 \
    --wandb-project neural-highlight \
    --wandb-name multilingual-streaming-comments-v18
