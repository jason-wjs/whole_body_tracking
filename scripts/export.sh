#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

uv run --project "${REPO_ROOT}" python -m whole_body_tracking.cli.export \
    --dataset-path /tmp/lafan1_compiled \
    --experiment-name g1_tracking \
    --load-run '.*lafan1_g1_single_gpu' \
    --output-dir /tmp/wbt_export
