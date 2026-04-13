"""Raw clip schema for canonical G1 `.npz` motion files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from whole_body_tracking.robot.g1 import G1_BODY_NAMES, G1_JOINT_NAMES

REQUIRED_NPZ_KEYS: tuple[str, ...] = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class RawClipInfo:
    path: Path
    keys: tuple[str, ...]
    num_frames: int
    num_joints: int
    num_bodies: int
    fps: int


def _assert_shape(name: str, value: np.ndarray, expected_rank: int) -> None:
    if value.ndim != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got shape {value.shape}")


def _assert_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _assert_unit_quaternion(name: str, value: np.ndarray, atol: float = 5e-2) -> None:
    norms = np.linalg.norm(value, axis=-1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError(f"{name} contains non-unit quaternions")


def validate_npz_payload(data: dict[str, np.ndarray], path: Path | None = None) -> RawClipInfo:
    keys = tuple(sorted(data.keys()))
    if set(REQUIRED_NPZ_KEYS) != set(keys):
        missing = sorted(set(REQUIRED_NPZ_KEYS) - set(keys))
        extra = sorted(set(keys) - set(REQUIRED_NPZ_KEYS))
        raise ValueError(f"Invalid keys for {path or '<memory>'}: missing={missing}, extra={extra}")

    joint_pos = np.asarray(data["joint_pos"])
    joint_vel = np.asarray(data["joint_vel"])
    body_pos_w = np.asarray(data["body_pos_w"])
    body_quat_w = np.asarray(data["body_quat_w"])
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"])
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"])
    fps_arr = np.asarray(data["fps"])

    _assert_shape("joint_pos", joint_pos, 2)
    _assert_shape("joint_vel", joint_vel, 2)
    _assert_shape("body_pos_w", body_pos_w, 3)
    _assert_shape("body_quat_w", body_quat_w, 3)
    _assert_shape("body_lin_vel_w", body_lin_vel_w, 3)
    _assert_shape("body_ang_vel_w", body_ang_vel_w, 3)
    if fps_arr.ndim != 1 or fps_arr.size != 1:
        raise ValueError(f"fps must have shape (1,), got {fps_arr.shape}")

    _assert_finite("joint_pos", joint_pos)
    _assert_finite("joint_vel", joint_vel)
    _assert_finite("body_pos_w", body_pos_w)
    _assert_finite("body_quat_w", body_quat_w)
    _assert_finite("body_lin_vel_w", body_lin_vel_w)
    _assert_finite("body_ang_vel_w", body_ang_vel_w)
    _assert_unit_quaternion("body_quat_w", body_quat_w)

    num_frames = int(joint_pos.shape[0])
    num_joints = int(joint_pos.shape[1])
    num_bodies = int(body_pos_w.shape[1])
    fps = int(round(float(fps_arr.reshape(-1)[0])))

    if joint_vel.shape != joint_pos.shape:
        raise ValueError("joint_vel must match joint_pos shape")
    if body_pos_w.shape != (num_frames, num_bodies, 3):
        raise ValueError("body_pos_w must have shape [T,B,3]")
    if body_quat_w.shape != (num_frames, num_bodies, 4):
        raise ValueError("body_quat_w must have shape [T,B,4]")
    if body_lin_vel_w.shape != (num_frames, num_bodies, 3):
        raise ValueError("body_lin_vel_w must have shape [T,B,3]")
    if body_ang_vel_w.shape != (num_frames, num_bodies, 3):
        raise ValueError("body_ang_vel_w must have shape [T,B,3]")
    if num_joints != len(G1_JOINT_NAMES):
        raise ValueError(f"Expected {len(G1_JOINT_NAMES)} joints, got {num_joints}")
    if num_bodies != len(G1_BODY_NAMES):
        raise ValueError(f"Expected {len(G1_BODY_NAMES)} bodies, got {num_bodies}")
    if num_frames <= 0:
        raise ValueError("motion clip must contain at least one frame")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    return RawClipInfo(
        path=path or Path("<memory>"),
        keys=tuple(REQUIRED_NPZ_KEYS),
        num_frames=num_frames,
        num_joints=num_joints,
        num_bodies=num_bodies,
        fps=fps,
    )


def validate_npz_file(path: Path) -> RawClipInfo:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    return validate_npz_payload(payload, path)
