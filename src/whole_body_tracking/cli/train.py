"""Training CLI for whole_body_tracking."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags

from whole_body_tracking.runtime.mjlab_guard import assert_official_mjlab_import
from whole_body_tracking.tracking.env import make_tracking_env
from whole_body_tracking.tracking.motion_source import MotionSource
from whole_body_tracking.tracking.rl_cfg import make_g1_tracking_ppo_runner_cfg
from whole_body_tracking.tracking.runner import WholeBodyTrackingOnPolicyRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whole_body_tracking.train")
    parser.add_argument("--dataset-path", action="append", required=True, help="Compiled dataset directory.")
    parser.add_argument(
        "--dataset-weight",
        action="append",
        type=float,
        help="Dataset sampling weights aligned with --dataset-path. Defaults to 1.0 per dataset.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--device", type=str, default=None, help="Execution device, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iterations", type=int, default=30_000, help="PPO iterations.")
    parser.add_argument("--num-steps-per-env", type=int, default=24, help="Rollout steps per PPO update.")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint save interval.")
    parser.add_argument(
        "--logger",
        choices=("tensorboard", "wandb"),
        default="tensorboard",
        help="Logger backend.",
    )
    parser.add_argument("--wandb-project", default="whole_body_tracking", help="W&B project name when logger=wandb.")
    parser.add_argument("--wandb-tag", action="append", default=None, help="Optional W&B tag. Repeatable.")
    parser.add_argument(
        "--upload-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether checkpoints and exported models should be uploaded when logger=wandb.",
    )
    parser.add_argument("--experiment-name", default="g1_tracking", help="Experiment name under logs/rsl_rl/.")
    parser.add_argument("--run-name", default="", help="Optional run name suffix.")
    parser.add_argument(
        "--sampling-mode",
        choices=("start", "uniform", "adaptive"),
        default="adaptive",
        help="Preflight MotionSource sampling mode used during dry-run.",
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
            sampling_mode=str(args.sampling_mode),
            seed=int(args.seed),
        ).reset()
        print("DRY RUN OK")
        return 0

    configure_torch_backends()
    device = _resolve_device(args.device)
    env = make_tracking_env(
        dataset_paths=dataset_paths,
        dataset_weights=dataset_weights,
        num_envs=int(args.num_envs),
        device=device,
        play=False,
        seed=int(args.seed),
    )
    runner_cfg = make_g1_tracking_ppo_runner_cfg(
        seed=int(args.seed),
        max_iterations=int(args.max_iterations),
        num_steps_per_env=int(args.num_steps_per_env),
        save_interval=int(args.save_interval),
        experiment_name=str(args.experiment_name),
        run_name=str(args.run_name),
        logger=str(args.logger),
        wandb_project=str(args.wandb_project),
        wandb_tags=tuple(args.wandb_tag or ()),
        upload_model=bool(args.upload_model),
    )
    log_root = Path("logs") / "rsl_rl" / runner_cfg.experiment_name
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if runner_cfg.run_name:
        log_dir_name += f"_{runner_cfg.run_name}"
    log_dir = log_root / log_dir_name
    vec_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    add_wandb_tags(runner_cfg.wandb_tags)
    runner = WholeBodyTrackingOnPolicyRunner(vec_env, asdict(runner_cfg), str(log_dir), device)
    runner.learn(
        num_learning_iterations=runner_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
