"""Tracking command utilities decoupled from mjlab's built-in task framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

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

from whole_body_tracking.tracking.motion_source import MotionSource

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class TrackingCommandConfig:
    anchor_body_name: str


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = q0.unbind(dim=-1)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dim=-1,
    )


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    qvec = quat[..., 1:]
    uv = torch.cross(qvec, vec, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vec + 2.0 * (quat[..., :1] * uv + uuv)


def _yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    half_yaw = yaw * 0.5
    out = torch.zeros_like(quat)
    out[..., 0] = torch.cos(half_yaw)
    out[..., 3] = torch.sin(half_yaw)
    return out


class TrackingCommand:
    def __init__(
        self,
        cfg: TrackingCommandConfig,
        *,
        motion_source: MotionSource,
        body_names: tuple[str, ...],
        joint_names: tuple[str, ...],
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.motion_source = motion_source
        self.body_names = tuple(body_names)
        self.joint_names = tuple(joint_names)
        self.device = device
        self.anchor_body_idx = self.body_names.index(cfg.anchor_body_name)

    def current_reference(self) -> dict[str, torch.Tensor]:
        view = self.motion_source.reference()
        return {
            key: torch.as_tensor(value[:, 0], dtype=torch.float32, device=self.device)
            for key, value in view.items()
        }

    def future_reference(self, frame_offsets: np.ndarray) -> dict[str, torch.Tensor]:
        view = self.motion_source.reference(frame_offsets=np.asarray(frame_offsets, dtype=np.int64))
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in view.items()
        }

    def relative_reference(
        self,
        robot_anchor_pos_w: torch.Tensor,
        robot_anchor_quat_w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        current = self.current_reference()
        anchor_pos_w = current["body_pos_w"][:, self.anchor_body_idx]
        anchor_quat_w = current["body_quat_w"][:, self.anchor_body_idx]

        anchor_pos_repeat = anchor_pos_w[:, None, :].expand(-1, len(self.body_names), -1)
        robot_anchor_pos_repeat = robot_anchor_pos_w[:, None, :].expand(-1, len(self.body_names), -1)
        delta_pos_w = robot_anchor_pos_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_repeat[..., 2]

        anchor_quat_repeat = anchor_quat_w[:, None, :].expand(-1, len(self.body_names), -1)
        robot_anchor_quat_repeat = robot_anchor_quat_w[:, None, :].expand(-1, len(self.body_names), -1)
        delta_ori_w = _yaw_quat(_quat_mul(robot_anchor_quat_repeat, _quat_conjugate(anchor_quat_repeat)))

        relative_body_quat_w = _quat_mul(delta_ori_w, current["body_quat_w"])
        relative_body_pos_w = delta_pos_w + _quat_apply(
            delta_ori_w,
            current["body_pos_w"] - anchor_pos_repeat,
        )

        out = dict(current)
        out["body_pos_w"] = relative_body_pos_w
        out["body_quat_w"] = relative_body_quat_w
        return out

    def record_failures(self, env_ids: np.ndarray) -> None:
        self.motion_source.record_failures(env_ids)

    def sampling_metrics(self) -> dict[str, torch.Tensor]:
        metrics = self.motion_source.sampling_metrics()
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in metrics.items()
        }


@dataclass(kw_only=True)
class MotionSourceCommandCfg(CommandTermCfg):
    dataset_paths: tuple[str, ...]
    dataset_weights: tuple[float, ...]
    anchor_body_name: str
    body_names: tuple[str, ...]
    entity_name: str
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

    def build(self, env: ManagerBasedRlEnv) -> "MotionSourceCommand":
        return MotionSourceCommand(self, env)


class MotionSourceCommand(CommandTerm):
    cfg: MotionSourceCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MotionSourceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.motion_source = MotionSource(
            dataset_paths=list(cfg.dataset_paths),
            dataset_path_weights=list(cfg.dataset_weights),
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
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )
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
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
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
            pose_ranges[:, 0], pose_ranges[:, 1], (env_ids.numel(), 6), device=self.device
        )
        root_pos[env_ids] += pose_delta[:, :3]
        root_ori[env_ids] = quat_mul(
            quat_from_euler_xyz(
                pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5]
            ),
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
            vel_ranges[:, 0], vel_ranges[:, 1], (env_ids.numel(), 6), device=self.device
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
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids],
            joint_vel[env_ids],
            env_ids=env_ids,
        )

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
        self.robot.clear_state(env_ids=env_ids)

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
