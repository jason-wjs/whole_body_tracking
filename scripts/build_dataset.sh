#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_DATASET_ROOT="${WBT_RAW_DATASET_ROOT:-/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz}"
COMPILED_DATASET_DIR="${WBT_COMPILED_DATASET_DIR:-/tmp/lafan1_compiled}"

uv run --project "${REPO_ROOT}" wbt-build-dataset \
  --dataset-root "${RAW_DATASET_ROOT}" \
  --output-dir "${COMPILED_DATASET_DIR}" \
  "$@"
