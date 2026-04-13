#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

## Validate a dataset
uv run --project "${REPO_ROOT}" python -m whole_body_tracking.cli.validate_dataset \
    /home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz

## Validate a single motion clip.
# uv run --project "${REPO_ROOT}" python -m whole_body_tracking.cli.validate_dataset \
#     /home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz/walk1_subject1.npz