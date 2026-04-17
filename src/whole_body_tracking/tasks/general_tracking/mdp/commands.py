"""Command term for local compiled-dataset motion tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

from .motion_source import MotionSource


@dataclass(kw_only=True)
class GeneralTrackingCommandCfg(CommandTermCfg):
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

  def build(self, env) -> "GeneralTrackingCommand":
    return GeneralTrackingCommand(self, env)


class GeneralTrackingCommand(CommandTerm):
  cfg: GeneralTrackingCommandCfg

  def __init__(self, cfg: GeneralTrackingCommandCfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    dataset_weights = cfg.dataset_weights
    if len(dataset_weights) == 0:
      dataset_weights = tuple(1.0 for _ in cfg.dataset_paths)
    if len(dataset_weights) != len(cfg.dataset_paths):
      raise ValueError("dataset_weights must be empty or match dataset_paths in length")

    self.motion_source = MotionSource(
      dataset_paths=list(cfg.dataset_paths),
      dataset_path_weights=list(dataset_weights),
      env_size=self.num_envs,
      sampling_mode=cfg.sampling_mode,
      adaptive_kernel_size=cfg.adaptive_kernel_size,
      adaptive_lambda=cfg.adaptive_lambda,
      adaptive_uniform_ratio=cfg.adaptive_uniform_ratio,
      adaptive_alpha=cfg.adaptive_alpha,
    )
    num_bodies = len(cfg.body_names)
    num_joints = len(self.robot.joint_names)
    self._joint_pos = torch.zeros((self.num_envs, num_joints), dtype=torch.float32, device=self.device)
    self._joint_vel = torch.zeros((self.num_envs, num_joints), dtype=torch.float32, device=self.device)
    self._body_pos_local_w = torch.zeros((self.num_envs, num_bodies, 3), dtype=torch.float32, device=self.device)
    self._body_quat_w = torch.zeros((self.num_envs, num_bodies, 4), dtype=torch.float32, device=self.device)
    self._body_quat_w[..., 0] = 1.0
    self._body_lin_vel_w = torch.zeros((self.num_envs, num_bodies, 3), dtype=torch.float32, device=self.device)
    self._body_ang_vel_w = torch.zeros((self.num_envs, num_bodies, 3), dtype=torch.float32, device=self.device)
    self.body_pos_relative_w = torch.zeros((self.num_envs, num_bodies, 3), dtype=torch.float32, device=self.device)
    self.body_quat_relative_w = torch.zeros((self.num_envs, num_bodies, 4), dtype=torch.float32, device=self.device)
    self.body_quat_relative_w[..., 0] = 1.0

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
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def time_steps(self) -> torch.Tensor:
    return torch.as_tensor(
      self.motion_source.active_frame_idx,
      dtype=torch.long,
      device=self.device,
    )

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._joint_pos

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._joint_vel

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_local_w + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _refresh_reference_cache(self) -> None:
    view = self.motion_source.reference()
    self._joint_pos = torch.as_tensor(view["joint_pos"][:, 0], dtype=torch.float32, device=self.device)
    self._joint_vel = torch.as_tensor(view["joint_vel"][:, 0], dtype=torch.float32, device=self.device)
    self._body_pos_local_w = torch.as_tensor(view["body_pos_w"][:, 0], dtype=torch.float32, device=self.device)
    self._body_quat_w = torch.as_tensor(view["body_quat_w"][:, 0], dtype=torch.float32, device=self.device)
    self._body_lin_vel_w = torch.as_tensor(view["body_lin_vel_w"][:, 0], dtype=torch.float32, device=self.device)
    self._body_ang_vel_w = torch.as_tensor(view["body_ang_vel_w"][:, 0], dtype=torch.float32, device=self.device)

  def _update_relative_reference(self) -> None:
    num_bodies = len(self.cfg.body_names)
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

    delta_pos_w = robot_anchor_pos_w_repeat.clone()
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))
    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w,
      self.body_pos_w - anchor_pos_w_repeat,
    )

  def _update_sampling_metrics(self) -> None:
    sampling_metrics = self.motion_source.sampling_metrics()
    for key, value in sampling_metrics.items():
      self.metrics[key][:] = float(value)

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w,
      dim=-1,
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w,
      self.robot_anchor_quat_w,
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w,
      dim=-1,
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w,
      dim=-1,
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w,
      dim=-1,
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w,
      self.robot_body_quat_w,
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w,
      dim=-1,
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w,
      dim=-1,
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos,
      dim=-1,
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel,
      dim=-1,
    )
    self._update_sampling_metrics()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    if self.cfg.sampling_mode == "adaptive":
      episode_failed = self._env.termination_manager.terminated[env_ids]
      if torch.any(episode_failed):
        failed_env_ids = env_ids[episode_failed].detach().cpu().numpy()
        self.motion_source.record_failures(failed_env_ids)

    env_ids_np = env_ids.detach().cpu().numpy()
    self.motion_source.reset(env_ids_np)
    self._refresh_reference_cache()

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

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
    root_pos[env_ids] += pose_delta[:, :3]
    root_ori[env_ids] = quat_mul(
      quat_from_euler_xyz(pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5]),
      root_ori[env_ids],
    )

    vel_ranges = torch.tensor(
      [
        self.cfg.velocity_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      dtype=torch.float32,
      device=self.device,
    )
    vel_delta = sample_uniform(
      vel_ranges[:, 0],
      vel_ranges[:, 1],
      (env_ids.numel(), 6),
      device=self.device,
    )
    root_lin_vel[env_ids] += vel_delta[:, :3]
    root_ang_vel[env_ids] += vel_delta[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()
    joint_pos[env_ids] += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos[env_ids].shape,
      device=self.device,
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids],
      soft_joint_pos_limits[:, :, 0],
      soft_joint_pos_limits[:, :, 1],
    )
    self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _update_command(self) -> None:
    next_frame_idx = self.motion_source.active_frame_idx + 1
    ended = next_frame_idx >= self.motion_source.active_clip_num_frames
    if np.any(~ended):
      self.motion_source.active_frame_idx[~ended] = next_frame_idx[~ended]
    if np.any(ended):
      ended_env_ids = torch.as_tensor(
        np.nonzero(ended)[0],
        dtype=torch.long,
        device=self.device,
      )
      self._resample_command(ended_env_ids)
    self._refresh_reference_cache()
    self._update_relative_reference()
    self._update_sampling_metrics()
