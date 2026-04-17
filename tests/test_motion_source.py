from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.helpers import build_compiled_dataset_dir
from whole_body_tracking.tasks.general_tracking.mdp.motion_source import MotionSource


def test_motion_source_start_mode_starts_from_first_frame(tmp_path: Path) -> None:
    dataset_dir = build_compiled_dataset_dir(tmp_path, "single_clip", num_clips=1)
    source = MotionSource(
        dataset_paths=[str(dataset_dir)],
        dataset_path_weights=[1.0],
        env_size=4,
        sampling_mode="start",
        seed=7,
    )

    source.reset()
    assert np.all(source.active_frame_idx == 0)


def test_motion_source_uniform_sampling_respects_dataset_weights(tmp_path: Path) -> None:
    dataset_a = build_compiled_dataset_dir(tmp_path, "dataset_a", num_clips=2)
    dataset_b = build_compiled_dataset_dir(tmp_path, "dataset_b", num_clips=2)
    source = MotionSource(
        dataset_paths=[str(dataset_a), str(dataset_b)],
        dataset_path_weights=[1.0, 3.0],
        env_size=4096,
        sampling_mode="uniform",
        seed=13,
    )

    source.reset()
    counts = np.bincount(source.active_dataset_ids, minlength=2)
    ratio = counts[1] / counts[0]
    assert 2.5 < ratio < 3.5


def test_motion_source_adaptive_sampling_biases_toward_failed_bin(tmp_path: Path) -> None:
    dataset_dir = build_compiled_dataset_dir(tmp_path, "adaptive_single", num_clips=1, num_frames=100)
    source = MotionSource(
        dataset_paths=[str(dataset_dir)],
        dataset_path_weights=[1.0],
        env_size=1,
        sampling_mode="adaptive",
        seed=5,
        adaptive_alpha=1.0,
        adaptive_uniform_ratio=0.0,
        adaptive_kernel_size=1,
    )

    source.reset()
    source.active_clip_ids[0] = 0
    source.active_frame_idx[0] = 80
    source.record_failures(np.array([0], dtype=np.int64))

    sampled = []
    for _ in range(64):
        source.reset(np.array([0], dtype=np.int64))
        sampled.append(int(source.active_frame_idx[0]))

    sampled = np.asarray(sampled, dtype=np.int64)
    assert sampled.min() >= 0
    assert sampled.mean() > 40


def test_motion_source_future_reference_clamps_at_clip_end(tmp_path: Path) -> None:
    dataset_dir = build_compiled_dataset_dir(tmp_path, "future_single", num_clips=1, num_frames=8)
    source = MotionSource(
        dataset_paths=[str(dataset_dir)],
        dataset_path_weights=[1.0],
        env_size=1,
        sampling_mode="start",
        seed=1,
    )

    source.reset()
    source.active_clip_ids[0] = 0
    source.active_frame_idx[0] = source.active_clip_num_frames[0] - 1
    future = source.reference(
        np.array([0], dtype=np.int64),
        frame_offsets=np.array([0, 1, 2], dtype=np.int64),
    )
    assert future["joint_pos"].shape == (1, 3, 29)
