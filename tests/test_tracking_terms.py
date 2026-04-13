from __future__ import annotations

import torch

from whole_body_tracking.tracking.observations import motion_anchor_ori_b, motion_anchor_pos_b
from whole_body_tracking.tracking.rewards import (
    motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp,
    motion_relative_body_position_error_exp,
)
from whole_body_tracking.tracking.terminations import (
    bad_anchor_ori,
    bad_anchor_pos_z_only,
    bad_motion_body_pos_z_only,
)


def test_motion_anchor_pos_b_returns_zero_for_matching_anchor_frames() -> None:
    pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    out = motion_anchor_pos_b(
        robot_anchor_pos_w=pos,
        robot_anchor_quat_w=quat,
        target_anchor_pos_w=pos,
        target_anchor_quat_w=quat,
    )
    assert torch.allclose(out, torch.zeros((1, 3), dtype=torch.float32))


def test_motion_anchor_ori_b_returns_identity_6d_for_matching_anchor_frames() -> None:
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    out = motion_anchor_ori_b(
        robot_anchor_pos_w=torch.zeros((1, 3), dtype=torch.float32),
        robot_anchor_quat_w=quat,
        target_anchor_pos_w=torch.zeros((1, 3), dtype=torch.float32),
        target_anchor_quat_w=quat,
    )
    expected = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-5)


def test_tracking_rewards_are_one_when_reference_matches_robot_state() -> None:
    pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    body_pos = pos[:, None, :]
    assert torch.allclose(
        motion_global_anchor_position_error_exp(pos, pos, std=0.3),
        torch.ones((1,), dtype=torch.float32),
    )
    assert torch.allclose(
        motion_global_anchor_orientation_error_exp(quat, quat, std=0.4),
        torch.ones((1,), dtype=torch.float32),
    )
    assert torch.allclose(
        motion_relative_body_position_error_exp(body_pos, body_pos, std=0.3),
        torch.ones((1,), dtype=torch.float32),
    )


def test_bad_anchor_pos_z_only_ignores_xy_error() -> None:
    robot_anchor = torch.tensor([[10.0, -5.0, 1.0]], dtype=torch.float32)
    target_anchor = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    assert not bool(bad_anchor_pos_z_only(target_anchor, robot_anchor, threshold=0.25).item())
    robot_anchor[:, 2] += 0.5
    assert bool(bad_anchor_pos_z_only(target_anchor, robot_anchor, threshold=0.25).item())


def test_bad_anchor_ori_uses_projected_gravity_difference() -> None:
    gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    flipped = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    assert not bool(bad_anchor_ori(identity, identity, gravity, threshold=0.8).item())
    assert bool(bad_anchor_ori(identity, flipped, gravity, threshold=0.8).item())


def test_bad_motion_body_pos_z_only_flags_any_body_exceeding_threshold() -> None:
    target = torch.zeros((1, 2, 3), dtype=torch.float32)
    robot = target.clone()
    assert not bool(bad_motion_body_pos_z_only(target, robot, threshold=0.25).item())
    robot[:, 1, 2] = 0.5
    assert bool(bad_motion_body_pos_z_only(target, robot, threshold=0.25).item())
