"""Reward terms aligned with mjlab tracking semantics."""

from __future__ import annotations

import torch

from whole_body_tracking.tracking.math_utils import quat_error_magnitude


def motion_global_anchor_position_error_exp(
    target_anchor_pos_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    *,
    std: float,
) -> torch.Tensor:
    error = torch.square(target_anchor_pos_w - robot_anchor_pos_w).sum(dim=-1)
    return torch.exp(-error / (std**2))


def motion_global_anchor_orientation_error_exp(
    target_anchor_quat_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    *,
    std: float,
) -> torch.Tensor:
    error = quat_error_magnitude(target_anchor_quat_w, robot_anchor_quat_w) ** 2
    return torch.exp(-error / (std**2))


def motion_relative_body_position_error_exp(
    target_body_pos_relative_w: torch.Tensor,
    robot_body_pos_w: torch.Tensor,
    *,
    std: float,
) -> torch.Tensor:
    error = torch.square(target_body_pos_relative_w - robot_body_pos_w).sum(dim=-1)
    return torch.exp(-error.mean(dim=-1) / (std**2))
