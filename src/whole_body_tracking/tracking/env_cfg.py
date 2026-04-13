"""Tracking task defaults aligned with mjlab tracking semantics."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg,
)

from whole_body_tracking.robot.g1 import G1_BODY_NAMES
from whole_body_tracking.tracking.command import MotionSourceCommandCfg


def make_g1_tracking_env_config() -> dict:
    return {
        "sim": {
            "dt": 0.02,
            "physics_dt": 0.005,
            "decimation": 4,
            "episode_length_s": 10.0,
        },
        "action": {
            "scale": 0.5,
            "use_default_offset": True,
        },
        "command": {
            "sampling_mode": "adaptive",
            "anchor_body_name": "torso_link",
            "body_names": G1_BODY_NAMES,
            "future_steps": [1, 2, 4, 8],
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
            "joint_position_range": (-0.1, 0.1),
            "adaptive_kernel_size": 1,
            "adaptive_lambda": 0.8,
            "adaptive_uniform_ratio": 0.1,
            "adaptive_alpha": 0.001,
        },
        "reward": {
            "motion_global_anchor_pos": {"weight": 0.5, "std": 0.3},
            "motion_global_anchor_ori": {"weight": 0.5, "std": 0.4},
            "motion_body_pos": {"weight": 1.0, "std": 0.3},
            "motion_body_ori": {"weight": 1.0, "std": 0.4},
            "motion_body_lin_vel": {"weight": 1.0, "std": 1.0},
            "motion_body_ang_vel": {"weight": 1.0, "std": 3.14},
            "action_rate_l2": {"weight": -0.1},
            "joint_limit": {"weight": -10.0},
            "self_collisions": {"weight": -0.1},
        },
        "termination": {
            "anchor_pos_threshold": 0.25,
            "anchor_ori_threshold": 0.8,
            "ee_body_pos_threshold": 0.25,
        },
    }


def make_mjlab_g1_tracking_env_cfg(
    *,
    dataset_paths: list[str],
    dataset_weights: list[float],
    num_envs: int = 1,
    play: bool = False,
    has_state_estimation: bool = True,
    seed: int | None = None,
) -> ManagerBasedRlEnvCfg:
    base = make_g1_tracking_env_config()
    cfg = unitree_g1_flat_tracking_env_cfg(
        has_state_estimation=has_state_estimation,
        play=play,
    )
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = seed
    command_defaults = base["command"]
    cfg.commands["motion"] = MotionSourceCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=not play,
        dataset_paths=tuple(dataset_paths),
        dataset_weights=tuple(dataset_weights),
        anchor_body_name=str(command_defaults["anchor_body_name"]),
        body_names=tuple(command_defaults["body_names"]),
        pose_range={} if play else dict(command_defaults["pose_range"]),
        velocity_range={} if play else dict(command_defaults["velocity_range"]),
        joint_position_range=tuple(command_defaults["joint_position_range"]),
        adaptive_kernel_size=int(command_defaults["adaptive_kernel_size"]),
        adaptive_lambda=float(command_defaults["adaptive_lambda"]),
        adaptive_uniform_ratio=float(command_defaults["adaptive_uniform_ratio"]),
        adaptive_alpha=float(command_defaults["adaptive_alpha"]),
        sampling_mode="start" if play else str(command_defaults["sampling_mode"]),
    )
    return cfg
