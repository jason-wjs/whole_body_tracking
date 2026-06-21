"""Export a general_tracking policy from a local checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from whole_body_tracking.tasks.general_tracking.mdp.commands import (
  MultiMotionCommandCfg,
)

from ._checkpoints import resolve_local_checkpoint

DEFAULT_TASK_ID = "Mjlab-GeneralTracking-Flat-Unitree-G1"


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task_id", nargs="?", default=DEFAULT_TASK_ID)
  parser.add_argument("--dataset-path", action="append", required=True)
  parser.add_argument("--dataset-weight", action="append", type=float)
  parser.add_argument("--checkpoint-file", type=str, default=None)
  parser.add_argument("--experiment-name", default=None)
  parser.add_argument("--load-run", default=".*")
  parser.add_argument("--load-checkpoint", default="model_.*.pt")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--verbose", action="store_true")
  return parser


def _resolve_weights(dataset_paths: list[str], dataset_weights: list[float] | None) -> tuple[float, ...]:
  if dataset_weights is None:
    return tuple(1.0 for _ in dataset_paths)
  if len(dataset_weights) != len(dataset_paths):
    raise ValueError("--dataset-weight must match --dataset-path in length")
  return tuple(dataset_weights)


def main(argv: Sequence[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  configure_torch_backends()

  dataset_paths = list(args.dataset_path)
  dataset_weights = _resolve_weights(dataset_paths, args.dataset_weight)

  import mjlab.tasks  # noqa: F401

  env_cfg = load_env_cfg(args.task_id, play=True)
  agent_cfg = load_rl_cfg(args.task_id)
  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MultiMotionCommandCfg)
  motion_cfg.dataset_paths = tuple(dataset_paths)
  motion_cfg.dataset_weights = dataset_weights
  env_cfg.scene.num_envs = 1

  checkpoint_path = resolve_local_checkpoint(
    experiment_name=args.experiment_name or agent_cfg.experiment_name,
    checkpoint_file=args.checkpoint_file,
    load_run=args.load_run,
    load_checkpoint=args.load_checkpoint,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=str(args.device))
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=str(args.device))
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=str(args.device),
  )
  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  runner.export_policy_to_onnx(str(output_dir), "policy.onnx", verbose=bool(args.verbose))
  runner.export_policy_to_jit(str(output_dir), "policy.pt")
  vec_env.close()
  print(f"EXPORT OK checkpoint={checkpoint_path} output_dir={output_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
