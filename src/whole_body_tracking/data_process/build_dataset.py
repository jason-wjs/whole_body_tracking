"""Build clip-preserving compiled datasets from canonical G1-ready `.npz` clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from whole_body_tracking.data_process.compiled_dataset import ARRAY_KEYS
from whole_body_tracking.data_process.schema import validate_npz_file
from whole_body_tracking.robot.g1 import G1_BODY_NAMES, G1_JOINT_NAMES


@dataclass(frozen=True)
class BuildDatasetSummary:
    output_dir: Path
    num_clips: int
    total_frames: int
    fps: int
    storage_dtype: str


def resolve_npz_paths(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    if root.is_file() and root.suffix == ".npz":
        return [root]
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"No .npz clips found in {dataset_root}")
    return paths


def build_compiled_dataset(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    target_fps: int = 50,
    storage_dtype: str = "float32",
) -> BuildDatasetSummary:
    clip_paths = resolve_npz_paths(dataset_root)
    infos = [validate_npz_file(path) for path in clip_paths]
    for info in infos:
        if info.fps != target_fps:
            raise ValueError(f"Expected fps={target_fps}, got {info.fps} for {info.path}")

    total_frames = int(sum(info.num_frames for info in infos))
    np_dtype = np.float16 if storage_dtype == "float16" else np.float32
    arrays = {
        "joint_pos": np.zeros((total_frames, len(G1_JOINT_NAMES)), dtype=np_dtype),
        "joint_vel": np.zeros((total_frames, len(G1_JOINT_NAMES)), dtype=np_dtype),
        "body_pos_w": np.zeros((total_frames, len(G1_BODY_NAMES), 3), dtype=np_dtype),
        "body_quat_w": np.zeros((total_frames, len(G1_BODY_NAMES), 4), dtype=np_dtype),
        "body_lin_vel_w": np.zeros((total_frames, len(G1_BODY_NAMES), 3), dtype=np_dtype),
        "body_ang_vel_w": np.zeros((total_frames, len(G1_BODY_NAMES), 3), dtype=np_dtype),
    }

    clip_frame_starts: list[int] = []
    clip_num_frames: list[int] = []
    labels: list[dict[str, int | str]] = []

    cursor = 0
    for clip_id, (path, info) in enumerate(zip(clip_paths, infos, strict=True)):
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        start = cursor
        end = cursor + info.num_frames
        span = slice(start, end)
        for key in ARRAY_KEYS:
            arrays[key][span] = payload[key]
        clip_frame_starts.append(start)
        clip_num_frames.append(info.num_frames)
        labels.append({"clip_id": clip_id, "source_path": str(path)})
        cursor = end

    output_dir = Path(output_dir)
    arrays_root = output_dir / "arrays"
    arrays_root.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        np.save(arrays_root / f"{key}.npy", value, allow_pickle=False)

    meta = {
        "dataset_name": output_dir.name,
        "fps": target_fps,
        "joint_names": list(G1_JOINT_NAMES),
        "body_names": list(G1_BODY_NAMES),
        "num_clips": len(clip_paths),
        "total_frames": total_frames,
        "clip_frame_starts": clip_frame_starts,
        "clip_num_frames": clip_num_frames,
        "clip_weights": [1.0] * len(clip_paths),
    }
    with (output_dir / "meta_motion.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True)
    with (output_dir / "id_label.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=True)

    return BuildDatasetSummary(
        output_dir=output_dir.resolve(),
        num_clips=len(clip_paths),
        total_frames=total_frames,
        fps=target_fps,
        storage_dtype=storage_dtype,
    )
