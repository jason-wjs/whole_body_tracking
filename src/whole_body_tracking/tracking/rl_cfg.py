"""PPO defaults aligned with mjlab tracking."""

from __future__ import annotations

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def make_g1_tracking_ppo_runner_cfg(
    *,
    seed: int = 42,
    max_iterations: int = 30_000,
    num_steps_per_env: int = 24,
    save_interval: int = 500,
    experiment_name: str = "g1_tracking",
    run_name: str = "",
    logger: str = "tensorboard",
    wandb_project: str = "whole_body_tracking",
    wandb_tags: tuple[str, ...] = (),
    upload_model: bool = True,
) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        run_name=run_name,
        logger=logger,
        wandb_project=wandb_project,
        wandb_tags=wandb_tags,
        upload_model=upload_model,
        save_interval=save_interval,
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
    )
