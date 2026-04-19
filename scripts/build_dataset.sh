#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

uv run --project "${REPO_ROOT}" wbt-build-dataset \
  --dataset-root "/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz" \
  --output-dir "/tmp/lafan1_compiled" \
  "$@"
