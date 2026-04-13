"""Play CLI for whole_body_tracking."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Sequence

import torch

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from whole_body_tracking.runtime.mjlab_guard import assert_official_mjlab_import
from whole_body_tracking.tracking.checkpoints import resolve_local_checkpoint
from whole_body_tracking.tracking.env import make_tracking_env
from whole_body_tracking.tracking.motion_source import MotionSource
from whole_body_tracking.tracking.rl_cfg import make_g1_tracking_ppo_runner_cfg
from whole_body_tracking.tracking.runner import WholeBodyTrackingOnPolicyRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whole_body_tracking.play")
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
        help="Agent type used during playback.",
    )
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Checkpoint path for --agent trained.")
    parser.add_argument("--experiment-name", default="g1_tracking", help="Experiment name under logs/rsl_rl/.")
    parser.add_argument("--load-run", default=".*", help="Regex used to select a local run directory when checkpoint is omitted.")
    parser.add_argument("--load-checkpoint", default="model_.*.pt", help="Regex used to select a local checkpoint when checkpoint is omitted.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--num-steps", type=int, default=None, help="Number of headless rollout steps when --viewer none.")
    parser.add_argument("--device", type=str, default=None, help="Execution device, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--viewer",
        choices=("auto", "none", "native", "viser"),
        default="auto",
        help="Viewer backend. Defaults to mjlab-like auto selection.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate dependencies and dataset config only.")
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


def _resolve_viewer(raw_viewer: str) -> str:
    if raw_viewer != "auto":
        return raw_viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


def _build_policy(agent: str, vec_env: RslRlVecEnvWrapper, checkpoint_file: str | None, device: str):
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    assert_official_mjlab_import()
    dataset_paths = list(args.dataset_path)
    dataset_weights = _resolve_weights(dataset_paths, args.dataset_weight)

    if args.dry_run:
        MotionSource(
            dataset_paths=dataset_paths,
            dataset_path_weights=dataset_weights,
            env_size=int(args.num_envs),
            sampling_mode="start",
            seed=int(args.seed),
        ).reset()
        print("DRY RUN OK")
        return 0

    configure_torch_backends()
    device = _resolve_device(args.device)
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
    env = make_tracking_env(
        dataset_paths=dataset_paths,
        dataset_weights=dataset_weights,
        num_envs=int(args.num_envs),
        device=device,
        play=True,
        seed=int(args.seed),
    )
    vec_env = RslRlVecEnvWrapper(env)
    policy = _build_policy(str(args.agent), vec_env, checkpoint_file, device)
    resolved_viewer = _resolve_viewer(str(args.viewer))
    if resolved_viewer == "native":
        NativeMujocoViewer(vec_env, policy).run()
        vec_env.close()
        return 0
    if resolved_viewer == "viser":
        ViserPlayViewer(vec_env, policy).run()
        vec_env.close()
        return 0
    if resolved_viewer != "none":
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
    if args.num_steps is None:
        raise ValueError("--num-steps is required when --viewer none")
    obs, _ = vec_env.reset()
    total_reward = torch.zeros(vec_env.num_envs, dtype=torch.float32, device=vec_env.device)
    for _ in range(int(args.num_steps)):
        actions = policy(obs)
        obs, reward, _dones, _extras = vec_env.step(actions)
        total_reward += reward
    print(f"PLAY OK mean_reward={float(total_reward.mean().item()):.6f}")
    vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
