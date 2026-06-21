from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers import build_compiled_dataset_dir
from whole_body_tracking.data.compiled_dataset import CompiledMotionDataset
from whole_body_tracking.data.g1_schema import G1_TRACKED_BODY_NAMES
from whole_body_tracking.tasks.general_tracking.mdp.commands import (
    MultiMotionCommandCfg,
    _CompiledMotionLoader,
)


def _set_clip_weights(dataset_dir: Path, weights: tuple[float, ...]) -> None:
    meta_path = dataset_dir / "meta_motion.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["clip_weights"] = list(weights)
    meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")


def test_compiled_motion_loader_concatenates_datasets_and_slices_tracked_bodies(
    tmp_path: Path,
) -> None:
    dataset_a_dir = build_compiled_dataset_dir(tmp_path, "dataset_a", num_clips=2, num_frames=5)
    dataset_b_dir = build_compiled_dataset_dir(tmp_path, "dataset_b", num_clips=3, num_frames=7)
    _set_clip_weights(dataset_a_dir, (1.0, 3.0))
    _set_clip_weights(dataset_b_dir, (2.0, 4.0, 6.0))

    dataset_a = CompiledMotionDataset.open(dataset_a_dir)
    dataset_b = CompiledMotionDataset.open(dataset_b_dir)
    tracked_body_indexes = [dataset_a.body_names.index(name) for name in G1_TRACKED_BODY_NAMES]

    loader = _CompiledMotionLoader(
        dataset_paths=(str(dataset_a_dir), str(dataset_b_dir)),
        dataset_weights=(2.0, 1.0),
        body_names=G1_TRACKED_BODY_NAMES,
        device="cpu",
    )

    assert loader.joint_names == dataset_a.joint_names
    assert loader.body_names == G1_TRACKED_BODY_NAMES
    assert loader.fps == dataset_a.fps
    assert loader.time_step_total == loader.joint_pos.shape[0]
    assert loader.time_step_total == dataset_a.total_frames + dataset_b.total_frames
    assert loader.joint_vel.shape == loader.joint_pos.shape
    assert loader.body_pos_w.shape == (
        loader.time_step_total,
        len(G1_TRACKED_BODY_NAMES),
        3,
    )
    assert loader.body_quat_w.shape == (
        loader.time_step_total,
        len(G1_TRACKED_BODY_NAMES),
        4,
    )
    assert loader.body_lin_vel_w.shape == loader.body_pos_w.shape
    assert loader.body_ang_vel_w.shape == loader.body_pos_w.shape

    torch.testing.assert_close(
        loader.body_pos_w[: dataset_a.total_frames],
        torch.as_tensor(dataset_a.arrays["body_pos_w"][:, tracked_body_indexes], dtype=torch.float32),
    )
    torch.testing.assert_close(
        loader.body_pos_w[dataset_a.total_frames :],
        torch.as_tensor(dataset_b.arrays["body_pos_w"][:, tracked_body_indexes], dtype=torch.float32),
    )

    expected_starts = torch.as_tensor(
        np.concatenate(
            [
                dataset_a.clip_frame_starts,
                dataset_b.clip_frame_starts + dataset_a.total_frames,
            ],
        ),
        dtype=torch.long,
    )
    expected_lengths = torch.as_tensor(
        np.concatenate([dataset_a.clip_num_frames, dataset_b.clip_num_frames]),
        dtype=torch.long,
    )
    torch.testing.assert_close(loader.clip_frame_starts, expected_starts)
    torch.testing.assert_close(loader.clip_num_frames, expected_lengths)
    torch.testing.assert_close(loader.clip_frame_ends, expected_starts + expected_lengths)

    expected_weights = torch.as_tensor(
        np.concatenate([2.0 * dataset_a.clip_weights, 1.0 * dataset_b.clip_weights]),
        dtype=torch.float32,
    )
    expected_weights = expected_weights / expected_weights.sum()
    torch.testing.assert_close(loader.clip_weights, expected_weights)
    torch.testing.assert_close(loader.clip_weights.sum(), torch.tensor(1.0))


@pytest.mark.parametrize(
    ("dataset_paths", "dataset_weights", "match"),
    [
        ((), (), "dataset_paths must be non-empty"),
        (("placeholder",), (1.0, 2.0), "dataset_weights must be empty or match dataset_paths"),
    ],
)
def test_compiled_motion_loader_rejects_invalid_path_and_weight_counts(
    tmp_path: Path,
    dataset_paths: tuple[str, ...],
    dataset_weights: tuple[float, ...],
    match: str,
) -> None:
    if dataset_paths:
        dataset_dir = build_compiled_dataset_dir(tmp_path, "dataset")
        dataset_paths = (str(dataset_dir),)

    with pytest.raises(ValueError, match=match):
        _CompiledMotionLoader(
            dataset_paths=dataset_paths,
            dataset_weights=dataset_weights,
            body_names=G1_TRACKED_BODY_NAMES,
            device="cpu",
        )


@pytest.mark.parametrize("dataset_weights", [(-1.0,), (0.0,)])
def test_compiled_motion_loader_rejects_non_positive_total_weights(
    tmp_path: Path,
    dataset_weights: tuple[float, ...],
) -> None:
    dataset_dir = build_compiled_dataset_dir(tmp_path, "dataset")

    with pytest.raises(ValueError, match="dataset_weights must be non-negative and sum to a positive value"):
        _CompiledMotionLoader(
            dataset_paths=(str(dataset_dir),),
            dataset_weights=dataset_weights,
            body_names=G1_TRACKED_BODY_NAMES,
            device="cpu",
        )


def test_multi_motion_command_cfg_defaults_dataset_and_body_fields() -> None:
    cfg = MultiMotionCommandCfg(entity_name="robot")

    assert cfg.dataset_paths == ()
    assert cfg.dataset_weights == ()
    assert cfg.anchor_body_name == ""
    assert cfg.body_names == ()
