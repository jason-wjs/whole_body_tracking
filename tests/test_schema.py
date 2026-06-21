from __future__ import annotations

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from whole_body_tracking.data.g1_schema import (
  G1_ANCHOR_BODY_NAME,
  G1_BODY_NAMES,
  G1_JOINT_NAMES,
  G1_TRACKED_BODY_NAMES,
)

EXPECTED_G1_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

EXPECTED_G1_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)

EXPECTED_G1_TRACKED_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def test_g1_schema_declares_stable_dataset_order() -> None:
  assert G1_JOINT_NAMES == EXPECTED_G1_JOINT_NAMES
  assert G1_BODY_NAMES == EXPECTED_G1_BODY_NAMES
  assert G1_ANCHOR_BODY_NAME == "torso_link"
  assert G1_TRACKED_BODY_NAMES == EXPECTED_G1_TRACKED_BODY_NAMES

  assert len(set(G1_JOINT_NAMES)) == len(G1_JOINT_NAMES)
  assert len(set(G1_BODY_NAMES)) == len(G1_BODY_NAMES)
  assert len(set(G1_TRACKED_BODY_NAMES)) == len(G1_TRACKED_BODY_NAMES)
  assert G1_ANCHOR_BODY_NAME in G1_BODY_NAMES
  assert G1_ANCHOR_BODY_NAME in G1_TRACKED_BODY_NAMES
  assert set(G1_TRACKED_BODY_NAMES) <= set(G1_BODY_NAMES)


def test_g1_schema_matches_current_mjlab_model_order() -> None:
  entity = Entity(get_g1_robot_cfg())

  assert G1_JOINT_NAMES == entity.joint_names
  assert G1_BODY_NAMES == entity.body_names


def test_g1_tracking_schema_matches_current_mjlab_tracking_config() -> None:
  cfg = unitree_g1_flat_tracking_env_cfg()
  motion_cmd = cfg.commands["motion"]

  assert G1_ANCHOR_BODY_NAME == motion_cmd.anchor_body_name
  assert G1_TRACKED_BODY_NAMES == tuple(motion_cmd.body_names)
