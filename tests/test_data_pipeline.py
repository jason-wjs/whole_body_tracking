from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.helpers import create_raw_dataset_root
from whole_body_tracking.data.build_dataset import build_compiled_dataset
from whole_body_tracking.data.compiled_dataset import ARRAY_KEYS, CompiledMotionDataset
from whole_body_tracking.data.schema import REQUIRED_NPZ_KEYS, validate_npz_file
from whole_body_tracking.robots.g1.schema import G1_BODY_NAMES, G1_JOINT_NAMES


def test_validate_npz_file_accepts_synthetic_clip(tmp_path: Path) -> None:
    raw_root = create_raw_dataset_root(tmp_path, num_clips=1)
    sample = next(raw_root.glob("*.npz"))
    info = validate_npz_file(sample)
    assert tuple(info.keys) == REQUIRED_NPZ_KEYS
    assert info.num_frames > 0
    assert info.num_joints == len(G1_JOINT_NAMES)
    assert info.num_bodies == len(G1_BODY_NAMES)
    assert info.fps == 50


def test_build_dataset_writes_expected_outputs(tmp_path: Path) -> None:
    raw_root = create_raw_dataset_root(tmp_path, num_clips=3)
    output_dir = tmp_path / "compiled"
    summary = build_compiled_dataset(dataset_root=raw_root, output_dir=output_dir)
    assert summary.num_clips == 3
    assert summary.total_frames > 0
    assert summary.fps == 50
    assert summary.storage_dtype == "float32"

    assert (output_dir / "meta_motion.json").is_file()
    assert (output_dir / "id_label.json").is_file()
    assert (output_dir / "arrays").is_dir()

    meta = json.loads((output_dir / "meta_motion.json").read_text(encoding="utf-8"))
    assert tuple(meta["joint_names"]) == G1_JOINT_NAMES
    assert tuple(meta["body_names"]) == G1_BODY_NAMES
    assert len(meta["clip_frame_starts"]) == len(meta["clip_num_frames"]) == 3
    assert meta["num_clips"] == 3
    assert meta["total_frames"] == sum(meta["clip_num_frames"])
    assert sorted((output_dir / "arrays").glob("*.npy"))


def test_compiled_dataset_can_load_and_slice_generated_output(tmp_path: Path) -> None:
    raw_root = create_raw_dataset_root(tmp_path, num_clips=2)
    output_dir = tmp_path / "compiled"
    build_compiled_dataset(dataset_root=raw_root, output_dir=output_dir)

    ds = CompiledMotionDataset.open(output_dir)
    assert ds.num_clips == 2
    assert ds.total_frames > 0
    assert ds.joint_names == G1_JOINT_NAMES
    assert ds.body_names == G1_BODY_NAMES
    assert sorted(ds.arrays.keys()) == sorted(ARRAY_KEYS)

    view = ds.get_frames(
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        steps=3,
    )
    assert view["joint_pos"].shape == (1, 3, len(G1_JOINT_NAMES))
    assert view["body_pos_w"].shape == (1, 3, len(G1_BODY_NAMES), 3)


def test_compiled_dataset_preserves_one_input_file_per_clip(tmp_path: Path) -> None:
    raw_root = create_raw_dataset_root(tmp_path, num_clips=4)
    output_dir = tmp_path / "compiled"
    build_compiled_dataset(dataset_root=raw_root, output_dir=output_dir)

    ds = CompiledMotionDataset.open(output_dir)
    labels = json.loads((output_dir / "id_label.json").read_text(encoding="utf-8"))
    source_files = sorted(raw_root.glob("*.npz"))

    assert ds.num_clips == len(source_files)
    assert len(labels) == len(source_files)
    assert {"clip_id", "source_path"} <= set(labels[0].keys())
