from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from whole_body_tracking.cli.build_dataset import main as build_dataset_main
from whole_body_tracking.cli.validate_dataset import main as validate_dataset_main
from whole_body_tracking.data_process.compiled_dataset import CompiledMotionDataset
from whole_body_tracking.data_process.schema import REQUIRED_NPZ_KEYS, validate_npz_file


LAFAN1_ROOT = Path("/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz")


def test_validate_npz_file_accepts_existing_lafan1_clip() -> None:
    sample = sorted(LAFAN1_ROOT.glob("*.npz"))[0]
    info = validate_npz_file(sample)
    assert tuple(info.keys) == REQUIRED_NPZ_KEYS
    assert info.num_frames > 0
    assert info.num_joints == 29
    assert info.num_bodies == 30
    assert info.fps == 50


def test_validate_dataset_cli_accepts_single_existing_clip(capsys) -> None:
    sample = sorted(LAFAN1_ROOT.glob("*.npz"))[0]
    rc = validate_dataset_main([str(sample)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "VALID" in out


def test_build_dataset_writes_expected_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "lafan1_compiled"
    rc = build_dataset_main(
        [
            "--dataset-root",
            str(LAFAN1_ROOT),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    assert (output_dir / "meta_motion.json").is_file()
    assert (output_dir / "id_label.json").is_file()
    assert (output_dir / "arrays").is_dir()

    with (output_dir / "meta_motion.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["fps"] == 50
    assert meta["joint_names"]
    assert meta["body_names"]
    assert len(meta["clip_frame_starts"]) == len(meta["clip_num_frames"]) > 0
    assert meta["num_clips"] == len(meta["clip_frame_starts"])
    assert meta["total_frames"] >= sum(meta["clip_num_frames"])
    assert sorted((output_dir / "arrays").glob("*.npy"))


def test_build_dataset_cli_prints_success_summary(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "lafan1_compiled"
    rc = build_dataset_main(
        [
            "--dataset-root",
            str(LAFAN1_ROOT),
            "--output-dir",
            str(output_dir),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "BUILD OK" in out
    assert f"output_dir={output_dir}" in out
    assert "clips=" in out
    assert "total_frames=" in out
    assert "fps=50" in out
    assert "dtype=float32" in out


def test_compiled_dataset_can_load_and_slice_generated_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "lafan1_compiled"
    build_dataset_main(
        [
            "--dataset-root",
            str(LAFAN1_ROOT),
            "--output-dir",
            str(output_dir),
        ]
    )

    ds = CompiledMotionDataset.open(output_dir)
    assert ds.num_clips > 0
    assert ds.total_frames > 0
    assert ds.joint_names
    assert ds.body_names

    view = ds.get_frames(
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        steps=3,
    )
    assert view["joint_pos"].shape == (1, 3, 29)
    assert view["body_pos_w"].shape == (1, 3, 30, 3)


def test_compiled_dataset_preserves_one_input_file_per_logical_clip(tmp_path: Path) -> None:
    output_dir = tmp_path / "lafan1_compiled"
    build_dataset_main(
        [
            "--dataset-root",
            str(LAFAN1_ROOT),
            "--output-dir",
            str(output_dir),
        ]
    )

    ds = CompiledMotionDataset.open(output_dir)
    labels = json.loads((output_dir / "id_label.json").read_text(encoding="utf-8"))
    sample_files = sorted(LAFAN1_ROOT.glob("*.npz"))

    assert ds.num_clips == len(sample_files)
    assert len(labels) == len(sample_files)
    assert {"clip_id", "source_path"} <= set(labels[0].keys())
