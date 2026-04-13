from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from whole_body_tracking.cli.build_dataset import main as build_dataset_main
from whole_body_tracking.tracking.command import TrackingCommand, TrackingCommandConfig
from whole_body_tracking.tracking.env_cfg import make_g1_tracking_env_config
from whole_body_tracking.tracking.motion_source import MotionSource


LAFAN1_ROOT = Path("/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz")


def _build_single_clip_dataset(tmp_path: Path) -> Path:
    sample = sorted(LAFAN1_ROOT.glob("*.npz"))[0]
    out_dir = tmp_path / "single_clip"
    rc = build_dataset_main(
        [
            "--dataset-root",
            str(sample),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    return out_dir


def test_make_g1_tracking_env_config_matches_expected_defaults() -> None:
    cfg = make_g1_tracking_env_config()
    assert cfg["action"]["scale"] == 0.5
    assert cfg["command"]["sampling_mode"] == "adaptive"
    assert cfg["command"]["anchor_body_name"] == "torso_link"
    assert cfg["sim"]["dt"] == 0.02
    assert cfg["reward"]["motion_body_pos"]["weight"] == 1.0
    assert cfg["termination"]["anchor_pos_threshold"] == 0.25


def test_tracking_command_relative_reference_aligns_robot_anchor_xy_and_yaw(tmp_path: Path) -> None:
    dataset_dir = _build_single_clip_dataset(tmp_path)
    source = MotionSource(
        dataset_paths=[str(dataset_dir)],
        dataset_path_weights=[1.0],
        env_size=1,
        sampling_mode="start",
        seed=3,
    )
    source.reset()
    command = TrackingCommand(
        TrackingCommandConfig(anchor_body_name="torso_link"),
        motion_source=source,
        body_names=source.body_names,
        joint_names=source.joint_names,
        device=torch.device("cpu"),
    )

    current = command.current_reference()
    robot_anchor_pos = current["body_pos_w"][:, command.anchor_body_idx].clone()
    robot_anchor_pos[:, 0] += 1.5
    robot_anchor_pos[:, 1] -= 0.7
    robot_anchor_quat = current["body_quat_w"][:, command.anchor_body_idx].clone()

    relative = command.relative_reference(robot_anchor_pos, robot_anchor_quat)
    relative_anchor = relative["body_pos_w"][:, command.anchor_body_idx]

    assert torch.allclose(relative_anchor[:, :2], robot_anchor_pos[:, :2], atol=1e-5)
    assert torch.allclose(
        relative_anchor[:, 2],
        current["body_pos_w"][:, command.anchor_body_idx, 2],
        atol=1e-5,
    )


def test_tracking_command_record_failures_updates_motion_source_metrics(tmp_path: Path) -> None:
    dataset_dir = _build_single_clip_dataset(tmp_path)
    source = MotionSource(
        dataset_paths=[str(dataset_dir)],
        dataset_path_weights=[1.0],
        env_size=1,
        sampling_mode="adaptive",
        seed=4,
        adaptive_alpha=1.0,
        adaptive_uniform_ratio=0.0,
        adaptive_kernel_size=1,
    )
    source.reset()
    source.active_frame_idx[0] = 1250
    command = TrackingCommand(
        TrackingCommandConfig(anchor_body_name="torso_link"),
        motion_source=source,
        body_names=source.body_names,
        joint_names=source.joint_names,
        device=torch.device("cpu"),
    )

    command.record_failures(np.array([0], dtype=np.int64))
    metrics = command.sampling_metrics()
    assert float(metrics["sampling_top1_prob"].item()) > 0.0
    assert 0.0 <= float(metrics["sampling_entropy"].item()) <= 1.0
