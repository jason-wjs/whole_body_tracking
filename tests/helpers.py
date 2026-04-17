from __future__ import annotations

from pathlib import Path

import numpy as np


def create_raw_clip(path: Path, *, num_frames: int = 16, fps: int = 50, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    joint_pos = rng.normal(size=(num_frames, 29)).astype(np.float32) * 0.05
    joint_vel = np.gradient(joint_pos, axis=0).astype(np.float32)

    body_pos_w = np.zeros((num_frames, 30, 3), dtype=np.float32)
    base_x = np.linspace(0.0, 0.05, num_frames, dtype=np.float32)
    for body_id in range(30):
        body_pos_w[:, body_id, 0] = base_x + body_id * 0.01
        body_pos_w[:, body_id, 1] = body_id * 0.005
        body_pos_w[:, body_id, 2] = 1.0 + body_id * 0.001

    body_quat_w = np.zeros((num_frames, 30, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.gradient(body_pos_w, axis=0).astype(np.float32)
    body_ang_vel_w = np.zeros((num_frames, 30, 3), dtype=np.float32)

    np.savez(
        path,
        fps=np.asarray([fps], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    return path


def create_raw_dataset_root(
    tmp_path: Path,
    *,
    num_clips: int = 2,
    num_frames: int = 16,
    fps: int = 50,
) -> Path:
    raw_root = tmp_path / "raw_clips"
    raw_root.mkdir(parents=True, exist_ok=True)
    for clip_id in range(num_clips):
        create_raw_clip(
            raw_root / f"clip_{clip_id}.npz",
            num_frames=num_frames + clip_id,
            fps=fps,
            seed=clip_id,
        )
    return raw_root


def build_compiled_dataset_dir(
    tmp_path: Path,
    name: str,
    *,
    num_clips: int = 2,
    num_frames: int = 16,
) -> Path:
    from whole_body_tracking.data.build_dataset import build_compiled_dataset

    raw_root = create_raw_dataset_root(
        tmp_path / f"{name}_raw",
        num_clips=num_clips,
        num_frames=num_frames,
    )
    output_dir = tmp_path / name
    build_compiled_dataset(dataset_root=raw_root, output_dir=output_dir)
    return output_dir
