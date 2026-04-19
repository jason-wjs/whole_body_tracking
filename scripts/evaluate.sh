#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

uv run --project "${REPO_ROOT}" wbt-evaluate "Mjlab-GeneralTracking-Flat-Unitree-G1" \
  --dataset-path "/tmp/lafan1_compiled" \
  --experiment-name "g1_general_tracking" \
  --load-run ".*lafan1_g1_single_gpu" \
  --device "cuda:0" \
  --num-envs "1024" \
  --output-file "/tmp/wbt_eval/metrics.json" \
  "$@"
