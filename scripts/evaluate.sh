#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TASK_ID="${WBT_TASK_ID:-Mjlab-GeneralTracking-Flat-Unitree-G1}"
COMPILED_DATASET_DIR="${WBT_COMPILED_DATASET_DIR:-/tmp/lafan1_compiled}"
EXPERIMENT_NAME="${WBT_EXPERIMENT_NAME:-g1_general_tracking}"
RUN_NAME="${WBT_RUN_NAME:-lafan1_g1_single_gpu}"
LOAD_RUN="${WBT_LOAD_RUN:-.*${RUN_NAME}}"
DEVICE="${WBT_DEVICE:-cuda:0}"
NUM_ENVS="${WBT_EVAL_NUM_ENVS:-1024}"
OUTPUT_FILE="${WBT_EVAL_OUTPUT_FILE:-/tmp/wbt_eval/metrics.json}"

uv run --project "${REPO_ROOT}" wbt-evaluate "${TASK_ID}" \
  --dataset-path "${COMPILED_DATASET_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --load-run "${LOAD_RUN}" \
  --device "${DEVICE}" \
  --num-envs "${NUM_ENVS}" \
  --output-file "${OUTPUT_FILE}" \
  "$@"
