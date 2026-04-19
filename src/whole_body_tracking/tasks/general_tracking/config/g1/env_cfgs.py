"""Unitree G1 environment configuration for general tracking."""

from __future__ import annotations

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from whole_body_tracking.tasks.general_tracking._dataset_env import parse_dataset_env
from whole_body_tracking.tasks.general_tracking.general_tracking_env_cfg import make_general_tracking_env_cfg
from whole_body_tracking.tasks.general_tracking.mdp.commands import GeneralTrackingCommandCfg


def unitree_g1_general_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  dataset_paths, dataset_weights = parse_dataset_env() if play else ((), ())
  cfg = make_general_tracking_env_cfg(
    play=play,
    has_state_estimation=has_state_estimation,
    dataset_paths=dataset_paths,
    dataset_weights=dataset_weights,
  )

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, GeneralTrackingCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"
  return cfg
