"""Simulator-backed tracking environment assembly using official mjlab runtime."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv

from whole_body_tracking.tracking.env_cfg import make_mjlab_g1_tracking_env_cfg


def make_tracking_env(
    *,
    dataset_paths: list[str],
    dataset_weights: list[float],
    num_envs: int = 1,
    device: str = "cpu",
    play: bool = False,
    has_state_estimation: bool = True,
    seed: int | None = None,
    render_mode: str | None = None,
) -> ManagerBasedRlEnv:
    cfg = make_mjlab_g1_tracking_env_cfg(
        dataset_paths=dataset_paths,
        dataset_weights=dataset_weights,
        num_envs=num_envs,
        play=play,
        has_state_estimation=has_state_estimation,
        seed=seed,
    )
    return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)
