# Copyright (c) 2024, Custom Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configuration for the UR5 reach task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UR5ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for UR5 reach task.

    Uses a [256, 128, 64] MLP with ELU activation for both actor and critic.
    Trains for 1500 iterations with checkpoints every 50 iterations.
    """

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "ur5_reach"
    run_name = ""
    logger = "tensorboard"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
