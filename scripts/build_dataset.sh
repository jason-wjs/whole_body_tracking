#!/usr/bin/env bash
set -euo pipefail

uv run wbt-build-dataset "$@"
