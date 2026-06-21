"""Evaluate a general_tracking policy with local compiled datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from whole_body_tracking.tasks.general_tracking.mdp.commands import (
  MultiMotionCommand,
  MultiMotionCommandCfg,
)
from whole_body_tracking.tasks.general_tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)

from ._checkpoints import resolve_local_checkpoint

DEFAULT_TASK_ID = "Mjlab-GeneralTracking-Flat-Unitree-G1"


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task_id", nargs="?", default=DEFAULT_TASK_ID)
  parser.add_argument("--dataset-path", action="append", required=True)
  parser.add_argument("--dataset-weight", action="append", type=float)
  parser.add_argument("--agent", choices=("zero", "random", "trained"), default="trained")
  parser.add_argument("--checkpoint-file", type=str, default=None)
  parser.add_argument("--experiment-name", default=None)
  parser.add_argument("--load-run", default=".*")
  parser.add_argument("--load-checkpoint", default="model_.*.pt")
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--num-steps", type=int, default=None)
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument("--output-file", type=str, default=None)
  return parser


def _resolve_weights(dataset_paths: list[str], dataset_weights: list[float] | None) -> tuple[float, ...]:
  if dataset_weights is None:
    return tuple(1.0 for _ in dataset_paths)
  if len(dataset_weights) != len(dataset_paths):
    raise ValueError("--dataset-weight must match --dataset-path in length")
  return tuple(dataset_weights)


def _resolve_device(raw_device: str | None) -> str:
  if raw_device is not None:
    return raw_device
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _build_policy(agent: str, vec_env: RslRlVecEnvWrapper, checkpoint_file: str | None, device: str, agent_cfg):
  if agent == "zero":
    def _policy(_obs):
      return torch.zeros((vec_env.num_envs, vec_env.num_actions), device=device)
    return _policy

  if agent == "random":
    def _policy(_obs):
      return 2.0 * torch.rand((vec_env.num_envs, vec_env.num_actions), device=device) - 1.0
    return _policy

  if checkpoint_file is None:
    raise ValueError("--checkpoint-file is required when --agent trained")
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=device)
  runner.load(
    checkpoint_file,
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def main(argv: Sequence[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  configure_torch_backends()

  dataset_paths = list(args.dataset_path)
  dataset_weights = _resolve_weights(dataset_paths, args.dataset_weight)
  device = _resolve_device(args.device)

  import mjlab.tasks  # noqa: F401

  env_cfg = load_env_cfg(args.task_id, play=False)
  agent_cfg = load_rl_cfg(args.task_id)
  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MultiMotionCommandCfg)
  motion_cfg.dataset_paths = tuple(dataset_paths)
  motion_cfg.dataset_weights = dataset_weights
  motion_cfg.sampling_mode = "start"
  env_cfg.scene.num_envs = int(args.num_envs)
  env_cfg.events.pop("push_robot", None)
  actor_obs_cfg = env_cfg.observations.get("actor")
  if actor_obs_cfg is not None:
    actor_obs_cfg.enable_corruption = True

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  checkpoint_file = args.checkpoint_file
  if args.agent == "trained":
    checkpoint_file = str(
      resolve_local_checkpoint(
        experiment_name=args.experiment_name or agent_cfg.experiment_name,
        checkpoint_file=args.checkpoint_file,
        load_run=args.load_run,
        load_checkpoint=args.load_checkpoint,
      )
    )
  policy = _build_policy(args.agent, vec_env, checkpoint_file, device, agent_cfg)

  command = vec_env.unwrapped.command_manager.get_term("motion")
  assert isinstance(command, MultiMotionCommand)
  ee_body_names = tuple(env_cfg.terminations["ee_body_pos"].params["body_names"])

  metric_stacks: dict[str, list[torch.Tensor]] = {
    "mpkpe": [],
    "r_mpkpe": [],
    "joint_vel_error": [],
    "ee_pos_error": [],
    "ee_ori_error": [],
  }
  done_envs = torch.zeros(vec_env.num_envs, dtype=torch.bool, device=vec_env.device)
  success = torch.zeros(vec_env.num_envs, dtype=torch.bool, device=vec_env.device)
  completed_episodes = 0

  obs, _ = vec_env.reset()
  step = 0
  while True:
    if args.num_steps is None and done_envs.all():
      break
    if args.num_steps is not None and step >= int(args.num_steps):
      break

    with torch.no_grad():
      actions = policy(obs)
    obs, _reward, dones, _extras = vec_env.step(actions)

    active = ~done_envs
    if active.any():
      metric_stacks["mpkpe"].append(torch.where(active, compute_mpkpe(command), 0.0))
      metric_stacks["r_mpkpe"].append(torch.where(active, compute_root_relative_mpkpe(command), 0.0))
      metric_stacks["joint_vel_error"].append(torch.where(active, compute_joint_velocity_error(command), 0.0))
      metric_stacks["ee_pos_error"].append(torch.where(active, compute_ee_position_error(command, ee_body_names), 0.0))
      metric_stacks["ee_ori_error"].append(torch.where(active, compute_ee_orientation_error(command, ee_body_names), 0.0))

    terminated = vec_env.unwrapped.termination_manager.terminated
    truncated = vec_env.unwrapped.termination_manager.time_outs
    newly_done = dones.bool() & ~done_envs
    if newly_done.any():
      success = success | (newly_done & truncated & ~terminated)
      done_envs = done_envs | newly_done
      completed_episodes += int(newly_done.sum().item())

    step += 1

  if metric_stacks["mpkpe"]:
    stacked = {key: torch.stack(values, dim=0) for key, values in metric_stacks.items()}
    active_steps = (stacked["mpkpe"] != 0).sum(dim=0).float().clamp(min=1)
    per_env_means = {key: values.sum(dim=0) / active_steps for key, values in stacked.items()}
    metrics = {
      "success_rate": float(success.float().mean().item()),
      "mpkpe": float(per_env_means["mpkpe"].mean().item()),
      "r_mpkpe": float(per_env_means["r_mpkpe"].mean().item()),
      "joint_vel_error": float(per_env_means["joint_vel_error"].mean().item()),
      "ee_pos_error": float(per_env_means["ee_pos_error"].mean().item()),
      "ee_ori_error": float(per_env_means["ee_ori_error"].mean().item()),
    }
  else:
    metrics = {
      "success_rate": 0.0,
      "mpkpe": 0.0,
      "r_mpkpe": 0.0,
      "joint_vel_error": 0.0,
      "ee_pos_error": 0.0,
      "ee_ori_error": 0.0,
    }
  metrics["completed_episodes"] = float(completed_episodes)
  metrics["num_envs"] = float(done_envs.numel())

  if args.output_file:
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

  vec_env.close()
  print(json.dumps(metrics, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
