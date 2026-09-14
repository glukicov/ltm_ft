#!/usr/bin/env bash
# Regenerate docker/requirements.txt from uv.lock: everything `ltm-ft` needs at runtime, minus torch/triton and the
# CUDA wheels, which come from the variant-specific PyTorch index in docker/finetune.Dockerfile instead.
set -euo pipefail
cd "$(dirname "$0")/.."
uv export --frozen --no-dev --no-emit-project --no-hashes \
  | grep -v -E '^\s*#' \
  | grep -v -E '^(torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==' > docker/requirements.txt
