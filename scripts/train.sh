#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TASK_ID="${WBT_TASK_ID:-Mjlab-GeneralTracking-Flat-Unitree-G1}"
COMPILED_DATASET_DIR="${WBT_COMPILED_DATASET_DIR:-/tmp/lafan1_compiled}"
DATASET_WEIGHT="${WBT_DATASET_WEIGHT:-1.0}"
NUM_ENVS="${WBT_NUM_ENVS:-4096}"
EXPERIMENT_NAME="${WBT_EXPERIMENT_NAME:-g1_general_tracking}"
RUN_NAME="${WBT_RUN_NAME:-lafan1_g1_single_gpu}"
MAX_ITERATIONS="${WBT_MAX_ITERATIONS:-100000}"

uv run --project "${REPO_ROOT}" train "${TASK_ID}" \
  --env.commands.motion.dataset-paths "('${COMPILED_DATASET_DIR}',)" \
  --env.commands.motion.dataset-weights "(${DATASET_WEIGHT},)" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.experiment-name "${EXPERIMENT_NAME}" \
  --agent.run-name "${RUN_NAME}" \
  --agent.max-iterations "${MAX_ITERATIONS}" \
  "$@"
