#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

uv run --project "${REPO_ROOT}" train "Mjlab-GeneralTracking-Flat-Unitree-G1" \
  --env.commands.motion.dataset-paths "('/tmp/lafan1_compiled',)" \
  --env.commands.motion.dataset-weights "(1.0,)" \
  --env.scene.num-envs "16384" \
  --agent.experiment-name "g1_general_tracking" \
  --agent.run-name "lafan1_g1_single_gpu" \
  --agent.max-iterations "30000" \
  "$@"
