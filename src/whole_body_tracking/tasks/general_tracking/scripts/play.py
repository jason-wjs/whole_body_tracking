"""wbt-play: inject compiled dataset paths before delegating to mjlab play."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from typing import Sequence

import torch

from whole_body_tracking.tasks.general_tracking._dataset_env import set_dataset_env


def _pop_repeatable_flag(argv: list[str], name: str) -> tuple[list[str], list[str]]:
  values: list[str] = []
  filtered: list[str] = []
  idx = 0
  while idx < len(argv):
    token = argv[idx]
    if token == name:
      if idx + 1 >= len(argv):
        raise ValueError(f"Missing value for {name}")
      values.append(argv[idx + 1])
      idx += 2
      continue
    prefix = f"{name}="
    if token.startswith(prefix):
      values.append(token[len(prefix) :])
      idx += 1
      continue
    filtered.append(token)
    idx += 1
  return values, filtered


def _delegate_to_mjlab_play(argv: list[str]) -> int:
  from mjlab.scripts.play import main as mjlab_play_main

  old_argv = sys.argv
  try:
    sys.argv = [old_argv[0], *argv]
    mjlab_play_main()
  finally:
    sys.argv = old_argv
  return 0


def _build_headless_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task_id")
  parser.add_argument("--agent", choices=("zero", "random", "trained"), default="trained")
  parser.add_argument("--checkpoint-file", type=str, default=None)
  parser.add_argument("--experiment-name", default=None)
  parser.add_argument("--load-run", default=".*")
  parser.add_argument("--load-checkpoint", default="model_.*.pt")
  parser.add_argument("--num-envs", type=int, default=None)
  parser.add_argument("--num-steps", type=int, required=True)
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument("--viewer", default="none")
  parser.add_argument("--no-terminations", action="store_true")
  return parser


def _resolve_device(raw_device: str | None) -> str:
  if raw_device is not None:
    return raw_device
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _run_headless_play(argv: list[str]) -> int:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  from ._checkpoints import resolve_local_checkpoint

  args = _build_headless_parser().parse_args(argv)
  configure_torch_backends()

  import mjlab.tasks  # noqa: F401

  device = _resolve_device(args.device)
  env_cfg = load_env_cfg(args.task_id, play=True)
  agent_cfg = load_rl_cfg(args.task_id)

  if args.no_terminations:
    env_cfg.terminations = {}
  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if args.agent == "zero":
    def policy(_obs):
      return torch.zeros((vec_env.num_envs, vec_env.num_actions), device=device)
  elif args.agent == "random":
    def policy(_obs):
      return 2.0 * torch.rand((vec_env.num_envs, vec_env.num_actions), device=device) - 1.0
  else:
    checkpoint_file = str(
      resolve_local_checkpoint(
        experiment_name=args.experiment_name or agent_cfg.experiment_name,
        checkpoint_file=args.checkpoint_file,
        load_run=args.load_run,
        load_checkpoint=args.load_checkpoint,
      )
    )
    runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=device)
    runner.load(
      checkpoint_file,
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

  obs, _ = vec_env.reset()
  total_reward = torch.zeros(vec_env.num_envs, dtype=torch.float32, device=vec_env.device)
  for _ in range(args.num_steps):
    actions = policy(obs)
    obs, reward, _dones, _extras = vec_env.step(actions)
    total_reward += reward
  print(f"PLAY OK mean_reward={float(total_reward.mean().item()):.6f}")
  vec_env.close()
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  args = list(sys.argv[1:] if argv is None else argv)
  dataset_paths, args = _pop_repeatable_flag(args, "--dataset-path")
  dataset_weights_raw, args = _pop_repeatable_flag(args, "--dataset-weight")
  dataset_weights = [float(value) for value in dataset_weights_raw]
  if dataset_weights and len(dataset_weights) != len(dataset_paths):
    raise ValueError("--dataset-weight must match --dataset-path in length")
  if dataset_paths:
    set_dataset_env(dataset_paths, dataset_weights or None)
  if "--viewer" in args:
    viewer_idx = args.index("--viewer")
    if viewer_idx + 1 < len(args) and args[viewer_idx + 1] == "none":
      return _run_headless_play(args)
  if any(token.startswith("--viewer=") and token.endswith("none") for token in args):
    return _run_headless_play(args)
  return _delegate_to_mjlab_play(args)


if __name__ == "__main__":
  raise SystemExit(main())
