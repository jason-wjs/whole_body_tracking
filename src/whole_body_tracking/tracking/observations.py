"""Observation terms aligned with mjlab tracking semantics."""

from __future__ import annotations

import torch

from whole_body_tracking.tracking.math_utils import matrix_from_quat, subtract_frame_transforms


def motion_anchor_pos_b(
    *,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    target_anchor_pos_w: torch.Tensor,
    target_anchor_quat_w: torch.Tensor,
) -> torch.Tensor:
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        target_anchor_pos_w,
        target_anchor_quat_w,
    )
    return pos_b


def motion_anchor_ori_b(
    *,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    target_anchor_pos_w: torch.Tensor,
    target_anchor_quat_w: torch.Tensor,
) -> torch.Tensor:
    _, quat_b = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        target_anchor_pos_w,
        target_anchor_quat_w,
    )
    mat = matrix_from_quat(quat_b)
    return mat[..., :2].reshape(mat.shape[0], -1)
