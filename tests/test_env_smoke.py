from __future__ import annotations

import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from tests.helpers import build_compiled_dataset_dir
from whole_body_tracking.data.g1_schema import (
    G1_ANCHOR_BODY_NAME,
    G1_TRACKED_BODY_NAMES,
)
from whole_body_tracking.tasks.general_tracking.config.g1.env_cfgs import (
    unitree_g1_general_tracking_env_cfg,
)
from whole_body_tracking.tasks.general_tracking.general_tracking_env_cfg import (
    make_general_tracking_env_cfg,
)
from whole_body_tracking.tasks.general_tracking.mdp.commands import (
    MultiMotionCommandCfg,
)


def test_base_general_tracking_cfg_leaves_robot_specific_body_names_unset() -> None:
    cfg = make_general_tracking_env_cfg()
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiMotionCommandCfg)
    assert motion_cfg.anchor_body_name == ""
    assert motion_cfg.body_names == ()


def test_play_env_cfg_reads_dataset_paths_from_environment(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = build_compiled_dataset_dir(tmp_path, "play_ds")
    monkeypatch.setenv("WBT_COMPILED_DATASET_PATHS", json.dumps([str(dataset_dir)]))
    monkeypatch.setenv("WBT_COMPILED_DATASET_WEIGHTS", json.dumps([1.0]))
    cfg = unitree_g1_general_tracking_env_cfg(play=True)
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiMotionCommandCfg)
    assert motion_cfg.anchor_body_name == G1_ANCHOR_BODY_NAME
    assert motion_cfg.body_names == G1_TRACKED_BODY_NAMES
    assert motion_cfg.dataset_paths == (str(dataset_dir),)
    assert motion_cfg.dataset_weights == (1.0,)
    assert motion_cfg.sampling_mode == "start"


def test_registered_env_cfg_can_reset_and_step_on_cpu(tmp_path: Path) -> None:
    dataset_dir = build_compiled_dataset_dir(tmp_path, "train_ds")
    cfg = unitree_g1_general_tracking_env_cfg(play=False)
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MultiMotionCommandCfg)
    assert motion_cfg.anchor_body_name == G1_ANCHOR_BODY_NAME
    assert motion_cfg.body_names == G1_TRACKED_BODY_NAMES
    motion_cfg.dataset_paths = (str(dataset_dir),)
    motion_cfg.dataset_weights = (1.0,)
    cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    obs, _extras = env.reset()
    assert set(obs.keys()) == {"actor", "critic"}
    action_dim = env.unwrapped.single_action_space.shape[0]
    action = torch.zeros((1, action_dim), dtype=torch.float32, device=env.device)
    obs, reward, terminated, timeouts, extras = env.step(action)
    assert obs["actor"].shape[0] == 1
    assert reward.shape == (1,)
    assert terminated.shape == (1,)
    assert timeouts.shape == (1,)
    assert isinstance(extras, dict)
    env.close()


def test_no_state_estimation_variant_removes_selected_actor_terms() -> None:
    cfg = unitree_g1_general_tracking_env_cfg(has_state_estimation=False)
    actor_terms = cfg.observations["actor"].terms
    assert "motion_anchor_pos_b" not in actor_terms
    assert "base_lin_vel" not in actor_terms
