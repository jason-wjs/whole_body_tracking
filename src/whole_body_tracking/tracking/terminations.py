"""Termination terms aligned with mjlab tracking semantics."""

from __future__ import annotations

import torch

from whole_body_tracking.tracking.math_utils import quat_apply_inverse


def bad_anchor_pos_z_only(
    target_anchor_pos_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    return (target_anchor_pos_w[:, 2] - robot_anchor_pos_w[:, 2]).abs() > threshold


def bad_anchor_ori(
    target_anchor_quat_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    gravity_vec_w: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    target_gravity_b = quat_apply_inverse(target_anchor_quat_w, gravity_vec_w)
    robot_gravity_b = quat_apply_inverse(robot_anchor_quat_w, gravity_vec_w)
    return (target_gravity_b[:, 2] - robot_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    target_body_pos_relative_w: torch.Tensor,
    robot_body_pos_w: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    error = (target_body_pos_relative_w[..., 2] - robot_body_pos_w[..., 2]).abs()
    return torch.any(error > threshold, dim=-1)
