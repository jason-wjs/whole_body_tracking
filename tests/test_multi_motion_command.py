from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.helpers import build_compiled_dataset_dir
from whole_body_tracking.data.compiled_dataset import CompiledMotionDataset
from whole_body_tracking.data.g1_schema import G1_TRACKED_BODY_NAMES
from whole_body_tracking.tasks.general_tracking.mdp.commands import (
    MultiMotionCommand,
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


@pytest.mark.parametrize("dataset_weights", [(-1.0,), (0.0,), (1.0, 0.0)])
def test_compiled_motion_loader_rejects_non_positive_total_weights(
    tmp_path: Path,
    dataset_weights: tuple[float, ...],
) -> None:
    dataset_dirs = [
        build_compiled_dataset_dir(tmp_path, f"dataset_{idx}")
        for idx in range(len(dataset_weights))
    ]

    with pytest.raises(ValueError, match="dataset_weights must be finite and positive"):
        _CompiledMotionLoader(
            dataset_paths=tuple(str(dataset_dir) for dataset_dir in dataset_dirs),
            dataset_weights=dataset_weights,
            body_names=G1_TRACKED_BODY_NAMES,
            device="cpu",
        )


@pytest.mark.parametrize("dataset_weights", [(float("inf"),), (float("nan"),)])
def test_compiled_motion_loader_rejects_non_finite_dataset_weights(
    tmp_path: Path,
    dataset_weights: tuple[float, ...],
) -> None:
    dataset_dir = build_compiled_dataset_dir(tmp_path, "dataset")

    with pytest.raises(ValueError, match="dataset_weights must be finite and positive"):
        _CompiledMotionLoader(
            dataset_paths=(str(dataset_dir),),
            dataset_weights=dataset_weights,
            body_names=G1_TRACKED_BODY_NAMES,
            device="cpu",
        )


@pytest.mark.parametrize("clip_weights", [(1.0, float("inf")), (1.0, float("nan"))])
def test_compiled_motion_loader_rejects_non_finite_clip_weights(
    tmp_path: Path,
    clip_weights: tuple[float, ...],
) -> None:
    dataset_dir = build_compiled_dataset_dir(tmp_path, "dataset", num_clips=2)
    _set_clip_weights(dataset_dir, clip_weights)

    with pytest.raises(ValueError, match="clip weights must be finite"):
        _CompiledMotionLoader(
            dataset_paths=(str(dataset_dir),),
            dataset_weights=(1.0,),
            body_names=G1_TRACKED_BODY_NAMES,
            device="cpu",
        )


def test_compiled_motion_loader_rejects_misaligned_clip_weights(tmp_path: Path) -> None:
    dataset_dir = build_compiled_dataset_dir(tmp_path, "dataset", num_clips=2)
    _set_clip_weights(dataset_dir, (1.0,))

    with pytest.raises(ValueError, match="clip weights must align with clip metadata"):
        _CompiledMotionLoader(
            dataset_paths=(str(dataset_dir),),
            dataset_weights=(1.0,),
            body_names=G1_TRACKED_BODY_NAMES,
            device="cpu",
        )


class _FakeGuiHandle:
    def __init__(self, value=0):
        self.value = value
        self.disabled = False
        self.callback = None

    def on_update(self, callback):
        self.callback = callback
        return callback

    def on_click(self, callback):
        self.callback = callback
        return callback


class _FakeGui:
    def __init__(self):
        self.slider = None
        self.checkbox = None
        self.button = None

    def add_folder(self, _name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return False

    def add_slider(self, _name, *, initial_value, **_kwargs):
        self.slider = _FakeGuiHandle(initial_value)
        return self.slider

    def add_checkbox(self, _name, *, initial_value):
        self.checkbox = _FakeGuiHandle(initial_value)
        return self.checkbox

    def add_button(self, _name):
        self.button = _FakeGuiHandle()
        return self.button


def _make_light_command(num_envs: int = 3) -> MultiMotionCommand:
    command = object.__new__(MultiMotionCommand)
    command._env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
        termination_manager=SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool)),
    )
    command.cfg = SimpleNamespace(adaptive_alpha=1.0)
    command.metrics = {
        "sampling_entropy": torch.zeros(num_envs),
        "sampling_top1_prob": torch.zeros(num_envs),
        "sampling_top1_bin": torch.zeros(num_envs),
    }
    command.motion = SimpleNamespace(
        time_step_total=16,
        clip_frame_starts=torch.tensor([0, 5, 12], dtype=torch.long),
        clip_num_frames=torch.tensor([5, 7, 4], dtype=torch.long),
        clip_frame_ends=torch.tensor([5, 12, 16], dtype=torch.long),
        clip_weights=torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32),
        joint_pos=torch.zeros(16, 2),
        joint_vel=torch.zeros(16, 2),
        body_pos_w=torch.zeros(16, 1, 3),
        body_quat_w=torch.zeros(16, 1, 4),
        body_lin_vel_w=torch.zeros(16, 1, 3),
        body_ang_vel_w=torch.zeros(16, 1, 3),
    )
    command.motion.body_quat_w[..., 0] = 1.0
    command.time_steps = torch.zeros(num_envs, dtype=torch.long)
    command._clip_ids = torch.zeros(num_envs, dtype=torch.long)
    command._clip_bin_counts = torch.tensor([2, 3, 2], dtype=torch.long)
    command._clip_bin_offsets = torch.tensor([0, 2, 5, 7], dtype=torch.long)
    command.bin_count = 7
    command.bin_failed_count = torch.zeros(command.bin_count, dtype=torch.float32)
    command._current_bin_failed = torch.zeros(command.bin_count, dtype=torch.float32)
    return command


def test_multi_motion_command_start_sampling_uses_clip_starts() -> None:
    command = _make_light_command()
    command._sample_clip_ids = lambda count: torch.tensor([2, 0], dtype=torch.long)
    env_ids = torch.tensor([0, 2], dtype=torch.long)

    command._start_sampling(env_ids)

    assert command._clip_ids.tolist() == [2, 0, 0]
    assert command.time_steps.tolist() == [12, 0, 0]


def test_multi_motion_command_uniform_sampling_stays_inside_selected_clips(monkeypatch) -> None:
    command = _make_light_command()
    command._sample_clip_ids = lambda count: torch.tensor([0, 1, 2], dtype=torch.long)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0, 0.999, 0.5]))

    command._uniform_sampling(torch.tensor([0, 1, 2], dtype=torch.long))

    starts = command.motion.clip_frame_starts[command._clip_ids]
    ends = command.motion.clip_frame_ends[command._clip_ids]
    assert torch.all(command.time_steps >= starts)
    assert torch.all(command.time_steps < ends)


def test_multi_motion_command_records_adaptive_failures_by_clip_bin() -> None:
    command = _make_light_command()
    command._env.termination_manager.terminated = torch.tensor([False, True, True])
    command._clip_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    command.time_steps = torch.tensor([0, 7, 15], dtype=torch.long)

    command._record_adaptive_failures(torch.tensor([0, 1, 2], dtype=torch.long))

    expected = torch.zeros(command.bin_count)
    expected[2] = 1.0
    expected[6] = 1.0
    torch.testing.assert_close(command.bin_failed_count, expected)


def test_multi_motion_command_update_resamples_only_ended_envs() -> None:
    command = _make_light_command(num_envs=4)
    command.time_steps = torch.tensor([3, 4, 8, 11], dtype=torch.long)
    command._clip_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    resampled = []

    def fake_resample(env_ids):
        resampled.append(env_ids.clone())
        command.time_steps[env_ids] = torch.tensor([0, 5], dtype=torch.long)
        command._clip_ids[env_ids] = torch.tensor([0, 1], dtype=torch.long)

    command._resample_command = fake_resample
    command.update_relative_body_poses = lambda: None

    command._update_command()

    assert len(resampled) == 1
    torch.testing.assert_close(resampled[0], torch.tensor([1, 3], dtype=torch.long))
    assert command.time_steps.tolist() == [4, 0, 9, 5]


def test_multi_motion_command_reset_to_frame_uses_global_frame_helper() -> None:
    command = _make_light_command(num_envs=2)
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    calls = []

    def fake_set_global_frames(called_env_ids, frames):
        calls.append((called_env_ids.clone(), frames.clone()))
        command.time_steps[called_env_ids] = torch.tensor([5, 5], dtype=torch.long)
        command._clip_ids[called_env_ids] = torch.tensor([1, 1], dtype=torch.long)

    command._set_global_frames = fake_set_global_frames
    command._write_reference_state_to_sim = lambda *args: None

    command.reset_to_frame(env_ids, 9)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], env_ids)
    torch.testing.assert_close(calls[0][1], torch.tensor([9, 9], dtype=torch.long))
    assert command.time_steps.tolist() == [5, 5]
    assert command._clip_ids.tolist() == [1, 1]


def test_multi_motion_command_gui_scrubber_syncs_clip_ids() -> None:
    command = object.__new__(MultiMotionCommand)
    command._env = SimpleNamespace(num_envs=1, device="cpu")
    command.motion = SimpleNamespace(
        time_step_total=10,
        clip_frame_ends=torch.tensor([5, 10], dtype=torch.long),
    )
    command.time_steps = torch.zeros(1, dtype=torch.long)
    command._clip_ids = torch.zeros(1, dtype=torch.long)
    gui = _FakeGui()
    server = SimpleNamespace(gui=gui)

    command.create_gui("motion", server, get_env_idx=lambda: 0)
    gui.slider.value = 7
    gui.slider.callback(None)

    assert command.time_steps.tolist() == [7]
    assert command._clip_ids.tolist() == [1]


def test_multi_motion_command_cfg_defaults_dataset_and_body_fields() -> None:
    cfg = MultiMotionCommandCfg(entity_name="robot")

    assert cfg.dataset_paths == ()
    assert cfg.dataset_weights == ()
    assert cfg.anchor_body_name == ""
    assert cfg.body_names == ()
