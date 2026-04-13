"""Compiled dataset format for clip-preserving G1 motion libraries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARRAY_KEYS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass
class CompiledMotionDataset:
    root: Path
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    fps: int
    clip_frame_starts: np.ndarray
    clip_num_frames: np.ndarray
    clip_weights: np.ndarray
    arrays: dict[str, np.ndarray]

    @property
    def num_clips(self) -> int:
        return int(self.clip_frame_starts.shape[0])

    @property
    def total_frames(self) -> int:
        return int(self.arrays["joint_pos"].shape[0])

    @classmethod
    def open(cls, root: str | Path) -> "CompiledMotionDataset":
        root = Path(root)
        with (root / "meta_motion.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        arrays_root = root / "arrays"
        arrays = {
            key: np.load(arrays_root / f"{key}.npy", mmap_mode="r")
            for key in ARRAY_KEYS
        }
        return cls(
            root=root,
            joint_names=tuple(meta["joint_names"]),
            body_names=tuple(meta["body_names"]),
            fps=int(meta["fps"]),
            clip_frame_starts=np.asarray(meta["clip_frame_starts"], dtype=np.int64),
            clip_num_frames=np.asarray(meta["clip_num_frames"], dtype=np.int64),
            clip_weights=np.asarray(meta.get("clip_weights", [1.0] * len(meta["clip_frame_starts"])), dtype=np.float64),
            arrays=arrays,
        )

    def get_frames(
        self,
        clip_ids: np.ndarray,
        local_starts: np.ndarray,
        *,
        steps: int | np.ndarray = 1,
    ) -> dict[str, np.ndarray]:
        clip_ids = np.asarray(clip_ids, dtype=np.int64).reshape(-1)
        local_starts = np.asarray(local_starts, dtype=np.int64).reshape(-1)
        if clip_ids.shape != local_starts.shape:
            raise ValueError("clip_ids and local_starts must have the same shape")

        if isinstance(steps, int):
            offsets = np.arange(steps, dtype=np.int64)[None, :]
        else:
            offsets = np.asarray(steps, dtype=np.int64).reshape(1, -1)

        starts = self.clip_frame_starts[clip_ids][:, None]
        lengths = self.clip_num_frames[clip_ids][:, None]
        ends = starts + np.clip(lengths - 1, 0, None)
        frame_ids = starts + local_starts[:, None] + offsets
        frame_ids = np.clip(frame_ids, starts, ends)

        out: dict[str, np.ndarray] = {}
        for key, value in self.arrays.items():
            out[key] = value[frame_ids]
        return out
