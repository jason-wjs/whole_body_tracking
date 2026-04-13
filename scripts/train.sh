#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

uv run --project "${REPO_ROOT}" python -m whole_body_tracking.cli.train \
    --logger wandb \
    --wandb-project whole_body_tracking \
    --run-name lafan1_g1_single_gpu \
    --dataset-path /tmp/lafan1_compiled \
    --device cuda:0 \
    --num-envs 4096 \
    --max-iterations 30000
