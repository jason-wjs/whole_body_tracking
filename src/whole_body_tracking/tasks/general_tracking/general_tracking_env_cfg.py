"""Base environment configuration for general tracking tasks."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from whole_body_tracking.data.g1_schema import G1_ANCHOR_BODY_NAME, G1_BODY_NAMES
from whole_body_tracking.tasks.general_tracking.mdp.commands import (
  GeneralTrackingCommandCfg,
)

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def make_general_tracking_env_cfg(
  *,
  play: bool = False,
  has_state_estimation: bool = True,
  dataset_paths: tuple[str, ...] = (),
  dataset_weights: tuple[float, ...] = (),
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  cfg.commands["motion"] = GeneralTrackingCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=not play,
    dataset_paths=dataset_paths,
    dataset_weights=dataset_weights,
    anchor_body_name=G1_ANCHOR_BODY_NAME,
    body_names=G1_BODY_NAMES,
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range=VELOCITY_RANGE,
    joint_position_range=(-0.1, 0.1),
    adaptive_kernel_size=1,
    adaptive_lambda=0.8,
    adaptive_uniform_ratio=0.1,
    adaptive_alpha=0.001,
    sampling_mode="adaptive",
  )

  if not has_state_estimation:
    new_actor_terms = {
      key: value
      for key, value in cfg.observations["actor"].terms.items()
      if key not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, GeneralTrackingCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
