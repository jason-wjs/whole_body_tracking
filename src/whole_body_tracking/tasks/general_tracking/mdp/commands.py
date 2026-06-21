"""Motion-command adapter for local compiled-dataset tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

from whole_body_tracking.data.compiled_dataset import ARRAY_KEYS, CompiledMotionDataset


class _CompiledMotionLoader:
  """Load and merge clip-preserving compiled motion datasets."""

  def __init__(
    self,
    *,
    dataset_paths: tuple[str, ...],
    dataset_weights: tuple[float, ...] = (),
    body_names: tuple[str, ...],
    device: str | torch.device = "cpu",
  ) -> None:
    if len(dataset_paths) == 0:
      raise ValueError("dataset_paths must be non-empty")

    if len(dataset_weights) == 0:
      dataset_weights = tuple(1.0 for _ in dataset_paths)
    if len(dataset_weights) != len(dataset_paths):
      raise ValueError("dataset_weights must be empty or match dataset_paths in length")

    dataset_weight_tensor = torch.tensor(dataset_weights, dtype=torch.float64)
    if not torch.all(torch.isfinite(dataset_weight_tensor)) or torch.any(dataset_weight_tensor <= 0):
      raise ValueError("dataset_weights must be finite and positive")

    datasets = [CompiledMotionDataset.open(path) for path in dataset_paths]
    joint_names = datasets[0].joint_names
    dataset_body_names = datasets[0].body_names
    fps = datasets[0].fps
    for dataset in datasets[1:]:
      if dataset.joint_names != joint_names:
        raise ValueError("All datasets must share joint_names")
      if dataset.body_names != dataset_body_names:
        raise ValueError("All datasets must share body_names")
      if dataset.fps != fps:
        raise ValueError("All datasets must share fps")

    try:
      body_indexes = [dataset_body_names.index(name) for name in body_names]
    except ValueError as exc:
      raise ValueError("body_names must exist in the compiled dataset body_names") from exc

    self.joint_names = joint_names
    self.body_names = tuple(body_names)
    self.fps = int(fps)

    tensor_kwargs = {"dtype": torch.float32, "device": device}
    self.joint_pos = self._concat_arrays(datasets, "joint_pos", **tensor_kwargs)
    self.joint_vel = self._concat_arrays(datasets, "joint_vel", **tensor_kwargs)
    self.body_pos_w = self._concat_body_arrays(datasets, "body_pos_w", body_indexes, **tensor_kwargs)
    self.body_quat_w = self._concat_body_arrays(datasets, "body_quat_w", body_indexes, **tensor_kwargs)
    self.body_lin_vel_w = self._concat_body_arrays(datasets, "body_lin_vel_w", body_indexes, **tensor_kwargs)
    self.body_ang_vel_w = self._concat_body_arrays(datasets, "body_ang_vel_w", body_indexes, **tensor_kwargs)
    self.time_step_total = int(self.joint_pos.shape[0])

    clip_frame_starts: list[np.ndarray] = []
    clip_num_frames: list[np.ndarray] = []
    weighted_clip_weights: list[np.ndarray] = []
    frame_offset = 0
    for dataset, dataset_weight in zip(datasets, dataset_weights, strict=True):
      clip_frame_starts.append(dataset.clip_frame_starts + frame_offset)
      clip_num_frames.append(dataset.clip_num_frames)
      weighted_clip_weights.append(np.asarray(dataset.clip_weights, dtype=np.float64) * float(dataset_weight))
      frame_offset += dataset.total_frames

    starts_np = np.concatenate(clip_frame_starts, axis=0).astype(np.int64, copy=False)
    lengths_np = np.concatenate(clip_num_frames, axis=0).astype(np.int64, copy=False)
    weights_np = np.concatenate(weighted_clip_weights, axis=0).astype(np.float64, copy=False)
    if weights_np.size == 0 or weights_np.shape != starts_np.shape or weights_np.shape != lengths_np.shape:
      raise ValueError("clip weights must align with clip metadata")
    if not np.all(np.isfinite(weights_np)):
      raise ValueError("clip weights must be finite")
    weight_sum = float(weights_np.sum())
    if np.any(weights_np < 0.0) or not np.isfinite(weight_sum) or weight_sum <= 0.0:
      raise ValueError("clip weights must be non-negative and sum to a positive finite value")

    self.clip_frame_starts = torch.as_tensor(starts_np, dtype=torch.long, device=device)
    self.clip_num_frames = torch.as_tensor(lengths_np, dtype=torch.long, device=device)
    self.clip_frame_ends = self.clip_frame_starts + self.clip_num_frames
    self.clip_weights = torch.as_tensor(weights_np / weights_np.sum(), dtype=torch.float32, device=device)

  def _concat_arrays(
    self,
    datasets: list[CompiledMotionDataset],
    key: str,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
  ) -> torch.Tensor:
    if key not in ARRAY_KEYS:
      raise ValueError(f"Unsupported compiled motion array: {key}")
    arrays = [np.asarray(dataset.arrays[key]) for dataset in datasets]
    return torch.as_tensor(np.concatenate(arrays, axis=0), dtype=dtype, device=device)

  def _concat_body_arrays(
    self,
    datasets: list[CompiledMotionDataset],
    key: str,
    body_indexes: list[int],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
  ) -> torch.Tensor:
    arrays = [np.asarray(dataset.arrays[key][:, body_indexes]) for dataset in datasets]
    return torch.as_tensor(np.concatenate(arrays, axis=0), dtype=dtype, device=device)


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  dataset_paths: tuple[str, ...] = ()
  dataset_weights: tuple[float, ...] = ()
  anchor_body_name: str = ""
  body_names: tuple[str, ...] = ()
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  viz: MotionCommandCfg.VizCfg = field(default_factory=MotionCommandCfg.VizCfg)

  def build(self, env) -> "MultiMotionCommand":
    return MultiMotionCommand(self, env)


class MultiMotionCommand(MotionCommand):
  cfg: MultiMotionCommandCfg

  def __init__(self, cfg: MultiMotionCommandCfg, env):
    CommandTerm.__init__(self, cfg, env)

    self.robot = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = _CompiledMotionLoader(
      dataset_paths=cfg.dataset_paths,
      dataset_weights=cfg.dataset_weights,
      body_names=cfg.body_names,
      device=self.device,
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs,
      len(cfg.body_names),
      3,
      dtype=torch.float32,
      device=self.device,
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs,
      len(cfg.body_names),
      4,
      dtype=torch.float32,
      device=self.device,
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self._clip_bin_counts = torch.maximum(
      torch.ones_like(self.motion.clip_num_frames),
      self.motion.clip_num_frames // max(self.motion.fps, 1) + 1,
    )
    self._clip_bin_offsets = torch.zeros(
      self.motion.clip_num_frames.shape[0] + 1,
      dtype=torch.long,
      device=self.device,
    )
    self._clip_bin_offsets[1:] = torch.cumsum(self._clip_bin_counts, dim=0)
    self.bin_count = int(self._clip_bin_offsets[-1].item())
    self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)
    self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)
    kernel_size = max(1, int(cfg.adaptive_kernel_size))
    self.kernel = torch.tensor(
      [float(cfg.adaptive_lambda) ** idx for idx in range(kernel_size)],
      dtype=torch.float32,
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    for name in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_body_lin_vel",
      "error_body_ang_vel",
      "error_joint_pos",
      "error_joint_vel",
      "sampling_entropy",
      "sampling_top1_prob",
      "sampling_top1_bin",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    if self.cfg.sampling_mode == "start":
      self._start_sampling(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._record_adaptive_failures(env_ids)
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    pose_ranges = torch.tensor(
      [
        self.cfg.pose_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      dtype=torch.float32,
      device=self.device,
    )
    pose_delta = sample_uniform(
      pose_ranges[:, 0],
      pose_ranges[:, 1],
      (env_ids.numel(), 6),
      device=self.device,
    )
    root_pos += pose_delta[:, :3]
    root_ori = quat_mul(
      quat_from_euler_xyz(pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5]),
      root_ori,
    )

    velocity_ranges = torch.tensor(
      [
        self.cfg.velocity_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      dtype=torch.float32,
      device=self.device,
    )
    velocity_delta = sample_uniform(
      velocity_ranges[:, 0],
      velocity_ranges[:, 1],
      (env_ids.numel(), 6),
      device=self.device,
    )
    root_lin_vel += velocity_delta[:, :3]
    root_ang_vel += velocity_delta[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]
    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=self.device,
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )

  def _update_command(self) -> None:
    next_time_steps = self.time_steps + 1
    clip_ends = self.motion.clip_frame_ends[self._clip_ids]
    ended = next_time_steps >= clip_ends
    active = ~ended
    self.time_steps[active] = next_time_steps[active]

    ended_env_ids = torch.where(ended)[0]
    if ended_env_ids.numel() > 0:
      self._resample_command(ended_env_ids)

    self.update_relative_body_poses()

  def _start_sampling(self, env_ids: torch.Tensor) -> None:
    clip_ids = self._sample_clip_ids(env_ids.numel())
    local_frames = torch.zeros(env_ids.numel(), dtype=torch.long, device=self.device)
    self._set_sampled_frames(env_ids, clip_ids, local_frames)
    self._update_sampling_metrics(self.motion.clip_weights)

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    clip_ids = self._sample_clip_ids(env_ids.numel())
    clip_lengths = self.motion.clip_num_frames[clip_ids].clamp_min(1)
    local_frames = torch.floor(torch.rand(env_ids.numel(), device=self.device) * clip_lengths.float()).long()
    self._set_sampled_frames(env_ids, clip_ids, local_frames)
    self._update_sampling_metrics(self._weighted_uniform_bin_probabilities())

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    probabilities = self._adaptive_bin_probabilities()
    sampled_bins = torch.multinomial(probabilities, env_ids.numel(), replacement=True)
    clip_ids = torch.searchsorted(self._clip_bin_offsets[1:], sampled_bins, right=True)
    local_bin_ids = sampled_bins - self._clip_bin_offsets[clip_ids]
    local_frames = self._sample_local_frames_from_bins(clip_ids, local_bin_ids)
    self._set_sampled_frames(env_ids, clip_ids, local_frames)
    self._update_sampling_metrics(probabilities)

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    if env_ids.numel() == 0:
      return
    frame_tensor = torch.full((env_ids.numel(),), frame, dtype=torch.long, device=self.device)
    self._set_global_frames(env_ids, frame_tensor)
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )

  def create_gui(self, name: str, server, get_env_idx, on_change=None, request_action=None) -> None:
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        env_ids = torch.tensor([idx], dtype=torch.long, device=self.device)
        frames = torch.tensor([int(scrubber.value)], dtype=torch.long, device=self.device)
        self._set_global_frames(env_ids, frames)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _sample_clip_ids(self, count: int) -> torch.Tensor:
    return torch.multinomial(self.motion.clip_weights, count, replacement=True)

  def _set_global_frames(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
    frames = torch.clamp(frames.long(), 0, self.motion.time_step_total - 1)
    clip_ids = torch.searchsorted(self.motion.clip_frame_ends, frames, right=True)
    clip_ids = torch.clamp(clip_ids, 0, self.motion.clip_frame_ends.shape[0] - 1)
    self._clip_ids[env_ids] = clip_ids
    self.time_steps[env_ids] = frames

  def _set_sampled_frames(
    self,
    env_ids: torch.Tensor,
    clip_ids: torch.Tensor,
    local_frames: torch.Tensor,
  ) -> None:
    clip_lengths = self.motion.clip_num_frames[clip_ids].clamp_min(1)
    local_frames = torch.minimum(local_frames.clamp_min(0), clip_lengths - 1)
    self._clip_ids[env_ids] = clip_ids
    self.time_steps[env_ids] = self.motion.clip_frame_starts[clip_ids] + local_frames

  def _record_adaptive_failures(self, env_ids: torch.Tensor) -> None:
    self._current_bin_failed.zero_()
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if not torch.any(episode_failed):
      return

    failed_env_ids = env_ids[episode_failed]
    clip_ids = self._clip_ids[failed_env_ids]
    local_frames = self.time_steps[failed_env_ids] - self.motion.clip_frame_starts[clip_ids]
    clip_lengths = self.motion.clip_num_frames[clip_ids].clamp_min(1)
    clip_bin_counts = self._clip_bin_counts[clip_ids].clamp_min(1)
    local_bins = torch.minimum(
      (local_frames * clip_bin_counts // clip_lengths).clamp_min(0),
      clip_bin_counts - 1,
    )
    flat_bins = self._clip_bin_offsets[clip_ids] + local_bins
    self._current_bin_failed[:] = torch.bincount(flat_bins, minlength=self.bin_count).to(self.device, torch.float32)
    self.bin_failed_count.mul_(1.0 - float(self.cfg.adaptive_alpha))
    self.bin_failed_count.add_(self._current_bin_failed, alpha=float(self.cfg.adaptive_alpha))

  def _weighted_uniform_bin_probabilities(self) -> torch.Tensor:
    probabilities = torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)
    for clip_id in range(self.motion.clip_weights.shape[0]):
      start = int(self._clip_bin_offsets[clip_id].item())
      end = int(self._clip_bin_offsets[clip_id + 1].item())
      probabilities[start:end] = self.motion.clip_weights[clip_id] / max(end - start, 1)
    return probabilities / probabilities.sum()

  def _adaptive_bin_probabilities(self) -> torch.Tensor:
    probabilities = torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)
    for clip_id in range(self.motion.clip_weights.shape[0]):
      start = int(self._clip_bin_offsets[clip_id].item())
      end = int(self._clip_bin_offsets[clip_id + 1].item())
      clip_probabilities = self.bin_failed_count[start:end].clone()
      clip_probabilities += float(self.cfg.adaptive_uniform_ratio) / max(end - start, 1)
      clip_probabilities = torch.clamp(clip_probabilities, min=0.0)
      if self.kernel.numel() > 1 and clip_probabilities.numel() > 0:
        padded = F.pad(
          clip_probabilities.view(1, 1, -1),
          (0, self.kernel.numel() - 1),
          mode="replicate",
        )
        clip_probabilities = F.conv1d(padded, self.kernel.view(1, 1, -1)).view(-1)
      probabilities[start:end] = clip_probabilities * self.motion.clip_weights[clip_id]

    total = probabilities.sum()
    if total <= 0.0:
      return self._weighted_uniform_bin_probabilities()
    return probabilities / total

  def _sample_local_frames_from_bins(
    self,
    clip_ids: torch.Tensor,
    local_bin_ids: torch.Tensor,
  ) -> torch.Tensor:
    clip_lengths = self.motion.clip_num_frames[clip_ids].clamp_min(1)
    clip_bin_counts = self._clip_bin_counts[clip_ids].clamp_min(1)
    max_frames = (clip_lengths - 1).clamp_min(0)
    bin_starts = torch.floor(local_bin_ids.float() * max_frames.float() / clip_bin_counts.float()).long()
    bin_ends = torch.floor((local_bin_ids.float() + 1.0) * max_frames.float() / clip_bin_counts.float()).long()
    frame_counts = (bin_ends - bin_starts + 1).clamp_min(1)
    offsets = torch.floor(torch.rand(clip_ids.numel(), device=self.device) * frame_counts.float()).long()
    return torch.minimum(bin_starts + offsets, max_frames)

  def _update_sampling_metrics(self, probabilities: torch.Tensor) -> None:
    probabilities = probabilities / probabilities.sum()
    entropy = -(probabilities * (probabilities + 1.0e-12).log()).sum()
    entropy_norm = entropy / math.log(max(int(probabilities.numel()), 2))
    top1_prob, top1_bin = probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = torch.clamp(entropy_norm, 0.0, 1.0)
    self.metrics["sampling_top1_prob"][:] = top1_prob
    self.metrics["sampling_top1_bin"][:] = top1_bin.float() / max(int(probabilities.numel()), 1)


GeneralTrackingCommandCfg = MultiMotionCommandCfg
GeneralTrackingCommand = MultiMotionCommand
