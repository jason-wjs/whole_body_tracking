"""Policy export CLI for whole_body_tracking."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from whole_body_tracking.runtime.mjlab_guard import assert_official_mjlab_import
from whole_body_tracking.tracking.checkpoints import resolve_local_checkpoint
from whole_body_tracking.tracking.env import make_tracking_env
from whole_body_tracking.tracking.rl_cfg import make_g1_tracking_ppo_runner_cfg
from whole_body_tracking.tracking.runner import WholeBodyTrackingOnPolicyRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whole_body_tracking.export")
    parser.add_argument("--dataset-path", action="append", required=True, help="Compiled dataset directory.")
    parser.add_argument(
        "--dataset-weight",
        action="append",
        type=float,
        help="Dataset sampling weights aligned with --dataset-path. Defaults to 1.0 per dataset.",
    )
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Checkpoint path. If omitted, resolve latest local checkpoint.")
    parser.add_argument("--experiment-name", default="g1_tracking", help="Experiment name under logs/rsl_rl/.")
    parser.add_argument("--load-run", default=".*", help="Regex used to select a local run directory when checkpoint is omitted.")
    parser.add_argument("--load-checkpoint", default="model_.*.pt", help="Regex used to select a local checkpoint when checkpoint is omitted.")
    parser.add_argument("--output-dir", required=True, help="Directory where exported policy artifacts are written.")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device used to materialize the policy.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose ONNX export logging.")
    return parser


def _resolve_weights(dataset_paths: list[str], dataset_weights: list[float] | None) -> list[float]:
    weights = dataset_weights or [1.0] * len(dataset_paths)
    if len(weights) != len(dataset_paths):
        raise ValueError("--dataset-weight must match --dataset-path in length")
    return list(weights)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    assert_official_mjlab_import()
    configure_torch_backends()
    dataset_paths = list(args.dataset_path)
    dataset_weights = _resolve_weights(dataset_paths, args.dataset_weight)
    checkpoint_path = resolve_local_checkpoint(
        experiment_name=str(args.experiment_name),
        checkpoint_file=args.checkpoint_file,
        load_run=str(args.load_run),
        load_checkpoint=str(args.load_checkpoint),
    )
    env = make_tracking_env(
        dataset_paths=dataset_paths,
        dataset_weights=dataset_weights,
        num_envs=1,
        device=str(args.device),
        play=True,
        seed=42,
    )
    vec_env = RslRlVecEnvWrapper(env)
    runner_cfg = make_g1_tracking_ppo_runner_cfg()
    runner = WholeBodyTrackingOnPolicyRunner(vec_env, asdict(runner_cfg), device=str(args.device))
    runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=str(args.device),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    runner.export_policy_artifacts(
        str(output_dir),
        onnx_filename="policy.onnx",
        jit_filename="policy.pt",
        verbose=bool(args.verbose),
    )
    vec_env.close()
    print(f"EXPORT OK checkpoint={checkpoint_path} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
