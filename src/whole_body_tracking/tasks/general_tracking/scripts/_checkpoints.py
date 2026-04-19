"""Local checkpoint resolution helpers."""

from __future__ import annotations

from pathlib import Path

from mjlab.utils.os import get_checkpoint_path


def resolve_local_checkpoint(
  *,
  experiment_name: str,
  checkpoint_file: str | None = None,
  load_run: str = ".*",
  load_checkpoint: str = "model_.*.pt",
  log_root: str | Path = "logs/rsl_rl",
) -> Path:
  if checkpoint_file is not None:
    path = Path(checkpoint_file).expanduser().resolve()
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path
  root = Path(log_root).expanduser().resolve() / experiment_name
  return get_checkpoint_path(root, load_run, load_checkpoint)
