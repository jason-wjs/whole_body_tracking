"""Evaluation CLI for whole_body_tracking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from whole_body_tracking.runtime.mjlab_guard import assert_official_mjlab_import
from whole_body_tracking.tracking.checkpoints import resolve_local_checkpoint
from whole_body_tracking.tracking.command import MotionSourceCommand
from whole_body_tracking.tracking.env_cfg import make_mjlab_g1_tracking_env_cfg
from whole_body_tracking.tracking.metrics import (
    compute_ee_orientation_error,
    compute_ee_position_error,
    compute_joint_velocity_error,
    compute_mpkpe,
    compute_root_relative_mpkpe,
)
from whole_body_tracking.tracking.rl_cfg import make_g1_tracking_ppo_runner_cfg
from whole_body_tracking.tracking.runner import WholeBodyTrackingOnPolicyRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whole_body_tracking.evaluate")
    parser.add_argument("--dataset-path", action="append", required=True, help="Compiled dataset directory.")
    parser.add_argument(
        "--dataset-weight",
        action="append",
        type=float,
        help="Dataset sampling weights aligned with --dataset-path. Defaults to 1.0 per dataset.",
    )
    parser.add_argument(
        "--agent",
        choices=("zero", "random", "trained"),
        default="trained",
        help="Policy used for evaluation.",
    )
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Checkpoint path for --agent trained.")
    parser.add_argument("--experiment-name", default="g1_tracking", help="Experiment name under logs/rsl_rl/.")
    parser.add_argument("--load-run", default=".*", help="Regex used to select a local run directory when checkpoint is omitted.")
    parser.add_argument("--load-checkpoint", default="model_.*.pt", help="Regex used to select a local checkpoint when checkpoint is omitted.")
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel evaluation environments.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Optional step cap for smoke tests. Defaults to running until all episodes finish.",
    )
    parser.add_argument("--device", type=str, default=None, help="Execution device, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional JSON file for evaluation metrics.")
    return parser


def _resolve_weights(dataset_paths: list[str], dataset_weights: list[float] | None) -> list[float]:
    weights = dataset_weights or [1.0] * len(dataset_paths)
    if len(weights) != len(dataset_paths):
        raise ValueError("--dataset-weight must match --dataset-path in length")
    return list(weights)


def _resolve_device(raw_device: str | None) -> str:
    if raw_device is not None:
        return raw_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_evaluate_env_cfg(
    *,
    dataset_paths: list[str],
    dataset_weights: list[float],
    num_envs: int,
    seed: int,
):
    cfg = make_mjlab_g1_tracking_env_cfg(
        dataset_paths=dataset_paths,
        dataset_weights=dataset_weights,
        num_envs=num_envs,
        play=False,
        seed=seed,
    )
    motion_cfg = cfg.commands["motion"]
    motion_cfg.sampling_mode = "start"
    cfg.events.pop("push_robot", None)
    # Match mjlab evaluate semantics: policy/actor observations stay corrupted.
    actor_obs_cfg = cfg.observations.get("actor")
    if actor_obs_cfg is not None:
        actor_obs_cfg.enable_corruption = True
    return cfg


def _build_policy(
    agent: str,
    vec_env: RslRlVecEnvWrapper,
    checkpoint_file: str | None,
    device: str,
):
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
    runner_cfg = make_g1_tracking_ppo_runner_cfg()
    runner = WholeBodyTrackingOnPolicyRunner(vec_env, asdict(runner_cfg), device=device)
    runner.load(
        checkpoint_file,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    return runner.get_inference_policy(device=device)


def _finalize_metrics(
    *,
    success: torch.Tensor,
    completed_episodes: int,
    done_envs: torch.Tensor,
    metric_stacks: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    if metric_stacks["mpkpe"]:
        stacked = {
            key: torch.stack(values, dim=0)
            for key, values in metric_stacks.items()
        }
        active_steps = (stacked["mpkpe"] != 0).sum(dim=0).float().clamp(min=1)
        per_env_means = {
            key: values.sum(dim=0) / active_steps
            for key, values in stacked.items()
        }
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
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    assert_official_mjlab_import()
    configure_torch_backends()

    dataset_paths = list(args.dataset_path)
    dataset_weights = _resolve_weights(dataset_paths, args.dataset_weight)
    device = _resolve_device(args.device)

    cfg = _make_evaluate_env_cfg(
        dataset_paths=dataset_paths,
        dataset_weights=dataset_weights,
        num_envs=int(args.num_envs),
        seed=int(args.seed),
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env)

    checkpoint_file = args.checkpoint_file
    if args.agent == "trained":
        checkpoint_file = str(
            resolve_local_checkpoint(
                experiment_name=str(args.experiment_name),
                checkpoint_file=args.checkpoint_file,
                load_run=str(args.load_run),
                load_checkpoint=str(args.load_checkpoint),
            )
        )
    policy = _build_policy(str(args.agent), vec_env, checkpoint_file, device)

    command = vec_env.unwrapped.command_manager.get_term("motion")
    assert isinstance(command, MotionSourceCommand)
    ee_body_names = tuple(cfg.terminations["ee_body_pos"].params["body_names"])

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
            metric_stacks["r_mpkpe"].append(
                torch.where(active, compute_root_relative_mpkpe(command), 0.0)
            )
            metric_stacks["joint_vel_error"].append(
                torch.where(active, compute_joint_velocity_error(command), 0.0)
            )
            metric_stacks["ee_pos_error"].append(
                torch.where(active, compute_ee_position_error(command, ee_body_names), 0.0)
            )
            metric_stacks["ee_ori_error"].append(
                torch.where(active, compute_ee_orientation_error(command, ee_body_names), 0.0)
            )

        terminated = vec_env.unwrapped.termination_manager.terminated
        truncated = vec_env.unwrapped.termination_manager.time_outs
        newly_done = dones.bool() & ~done_envs
        if newly_done.any():
            success = success | (newly_done & truncated & ~terminated)
            done_envs = done_envs | newly_done
            completed_episodes += int(newly_done.sum().item())

        step += 1

    metrics = _finalize_metrics(
        success=success,
        completed_episodes=completed_episodes,
        done_envs=done_envs,
        metric_stacks=metric_stacks,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    if args.output_file is not None:
        output_path = Path(args.output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
