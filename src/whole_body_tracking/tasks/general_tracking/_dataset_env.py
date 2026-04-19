"""Dataset path helpers shared by general_tracking configs and scripts."""

from __future__ import annotations

import json
import os

DATASET_PATHS_ENV = "WBT_COMPILED_DATASET_PATHS"
DATASET_WEIGHTS_ENV = "WBT_COMPILED_DATASET_WEIGHTS"


def parse_dataset_env() -> tuple[tuple[str, ...], tuple[float, ...]]:
  raw_paths = os.environ.get(DATASET_PATHS_ENV)
  raw_weights = os.environ.get(DATASET_WEIGHTS_ENV)
  if not raw_paths:
    return (), ()

  paths = tuple(str(path) for path in json.loads(raw_paths))
  if not paths:
    return (), ()

  if raw_weights:
    weights = tuple(float(weight) for weight in json.loads(raw_weights))
    if len(weights) != len(paths):
      raise ValueError("Dataset path and weight env vars must have the same length")
    return paths, weights

  return paths, tuple(1.0 for _ in paths)


def set_dataset_env(paths: list[str], weights: list[float] | None = None) -> None:
  os.environ[DATASET_PATHS_ENV] = json.dumps(paths)
  if weights:
    os.environ[DATASET_WEIGHTS_ENV] = json.dumps(weights)
  else:
    os.environ.pop(DATASET_WEIGHTS_ENV, None)
