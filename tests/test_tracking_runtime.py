from __future__ import annotations

from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.cli.build_dataset import main as build_dataset_main
from whole_body_tracking.cli.evaluate import build_parser as build_evaluate_parser
from whole_body_tracking.cli.evaluate import _make_evaluate_env_cfg
from whole_body_tracking.cli.evaluate import main as evaluate_main
from whole_body_tracking.cli.play import main as play_main
from whole_body_tracking.tracking.checkpoints import resolve_local_checkpoint
from whole_body_tracking.tracking.command import MotionSourceCommandCfg
from whole_body_tracking.tracking.env import make_tracking_env
from whole_body_tracking.tracking.env_cfg import make_mjlab_g1_tracking_env_cfg
from whole_body_tracking.tracking.rl_cfg import make_g1_tracking_ppo_runner_cfg


LAFAN1_ROOT = Path("/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz")


def _build_single_clip_dataset(tmp_path: Path) -> Path:
    sample = sorted(LAFAN1_ROOT.glob("*.npz"))[0]
    out_dir = tmp_path / "single_clip_runtime"
    rc = build_dataset_main(
        [
            "--dataset-root",
            str(sample),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    return out_dir


def test_make_g1_tracking_ppo_runner_cfg_matches_reference_defaults() -> None:
    cfg = make_g1_tracking_ppo_runner_cfg()
    assert isinstance(cfg, RslRlOnPolicyRunnerCfg)
    assert cfg.actor.hidden_dims == (512, 256, 128)
    assert cfg.critic.hidden_dims == (512, 256, 128)
    assert cfg.actor.activation == "elu"
    assert cfg.critic.activation == "elu"
    assert cfg.num_steps_per_env == 24
    assert cfg.max_iterations == 30_000
    assert cfg.algorithm.entropy_coef == 0.005
    assert cfg.algorithm.clip_param == 0.2


def test_make_g1_tracking_ppo_runner_cfg_accepts_wandb_project_and_tags() -> None:
    cfg = make_g1_tracking_ppo_runner_cfg(
        logger="wandb",
        wandb_project="my_project",
        wandb_tags=("tag_a", "tag_b"),
        upload_model=False,
    )
    assert cfg.logger == "wandb"
    assert cfg.wandb_project == "my_project"
    assert cfg.wandb_tags == ("tag_a", "tag_b")
    assert cfg.upload_model is False


def test_make_mjlab_g1_tracking_env_cfg_uses_motion_source_command(tmp_path: Path) -> None:
    dataset_dir = _build_single_clip_dataset(tmp_path)
    cfg = make_mjlab_g1_tracking_env_cfg(
        dataset_paths=[str(dataset_dir)],
        dataset_weights=[1.0],
        num_envs=1,
        play=False,
    )
    assert cfg.scene.num_envs == 1
    assert isinstance(cfg.commands["motion"], MotionSourceCommandCfg)
    motion_cfg = cfg.commands["motion"]
    assert motion_cfg.dataset_paths == (str(dataset_dir),)
    assert motion_cfg.dataset_weights == (1.0,)
    assert motion_cfg.anchor_body_name == "torso_link"
    assert motion_cfg.sampling_mode == "adaptive"


def test_make_tracking_env_can_reset_and_step_on_cpu(tmp_path: Path) -> None:
    dataset_dir = _build_single_clip_dataset(tmp_path)
    env = make_tracking_env(
        dataset_paths=[str(dataset_dir)],
        dataset_weights=[1.0],
        num_envs=1,
        device="cpu",
        play=False,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    obs, _extras = env.reset()
    assert set(obs.keys()) == {"actor", "critic"}
    assert obs["actor"].shape[0] == 1
    action_dim = env.unwrapped.single_action_space.shape[0]
    action = torch.zeros((1, action_dim), dtype=torch.float32, device=env.device)
    obs, reward, terminated, timeouts, extras = env.step(action)
    assert obs["actor"].shape[0] == 1
    assert reward.shape == (1,)
    assert terminated.shape == (1,)
    assert timeouts.shape == (1,)
    assert isinstance(extras, dict)
    env.close()


def test_play_cli_headless_zero_agent_smoke(tmp_path: Path) -> None:
    dataset_dir = _build_single_clip_dataset(tmp_path)
    rc = play_main(
        [
            "--dataset-path",
            str(dataset_dir),
            "--agent",
            "zero",
            "--viewer",
            "none",
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--num-steps",
            "2",
        ]
    )
    assert rc == 0


def test_play_cli_requires_num_steps_for_headless_mode(tmp_path: Path) -> None:
    dataset_dir = _build_single_clip_dataset(tmp_path)
    try:
        play_main(
            [
                "--dataset-path",
                str(dataset_dir),
                "--agent",
                "zero",
                "--viewer",
                "none",
                "--device",
                "cpu",
                "--num-envs",
                "1",
            ]
        )
    except ValueError as exc:
        assert "--num-steps is required when --viewer none" in str(exc)
    else:
        raise AssertionError("Expected headless play without --num-steps to fail")


def test_evaluate_cli_headless_zero_agent_smoke(tmp_path: Path) -> None:
    dataset_dir = _build_single_clip_dataset(tmp_path)
    output_file = tmp_path / "eval_metrics.json"
    rc = evaluate_main(
        [
            "--dataset-path",
            str(dataset_dir),
            "--agent",
            "zero",
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--num-steps",
            "2",
            "--output-file",
            str(output_file),
        ]
    )
    assert rc == 0
    assert output_file.is_file()


def test_evaluate_cli_default_num_envs_matches_mjlab_reference() -> None:
    parser = build_evaluate_parser()
    args = parser.parse_args(
        [
            "--dataset-path",
            "/tmp/fake_compiled",
        ]
    )
    assert args.num_envs == 1024


def test_evaluate_env_cfg_explicitly_enables_actor_corruption(tmp_path: Path) -> None:
    dataset_dir = _build_single_clip_dataset(tmp_path)
    cfg = _make_evaluate_env_cfg(
        dataset_paths=[str(dataset_dir)],
        dataset_weights=[1.0],
        num_envs=1,
        seed=42,
    )
    assert cfg.observations["actor"].enable_corruption is True


def test_resolve_local_checkpoint_picks_latest_matching_checkpoint(tmp_path: Path) -> None:
    exp_root = tmp_path / "logs" / "rsl_rl" / "g1_tracking"
    run_a = exp_root / "2026-04-13_10-00-00_run"
    run_b = exp_root / "2026-04-13_11-00-00_run"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "model_1.pt").write_bytes(b"a")
    (run_b / "model_2.pt").write_bytes(b"b")
    resolved = resolve_local_checkpoint(
        experiment_name="g1_tracking",
        log_root=tmp_path / "logs" / "rsl_rl",
    )
    assert resolved == (run_b / "model_2.pt").resolve()
