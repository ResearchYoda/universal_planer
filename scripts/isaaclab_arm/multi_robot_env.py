"""MultiRobotVecEnv: joint training across multiple robot morphologies.

Wraps multiple RslRlVecEnvWrapper instances and combines their rollouts into
a single batch so OnPolicyRunner trains one universal policy on all robots
simultaneously — not sequentially.

Key design decisions:
  - num_actions = MAX_JOINTS (8): policy always outputs 8 action dims.
    The env wrapper slices to each robot's actual DOF before stepping.
  - Observations are already uniform (78-dim) due to pGraph padding.
  - episode_length_buf is concatenated for reading, distributed for writing.
"""

import torch
from tensordict import TensorDict

from scripts.isaaclab_arm.pgraph import MAX_JOINTS


class MultiRobotVecEnv:
    """Joint multi-robot vectorized environment for universal policy training.

    Args:
        envs: Mapping of robot name → RslRlVecEnvWrapper.
              All envs must have the same obs dim (78) and device.
    """

    def __init__(self, envs: dict):
        self.envs = envs
        self.robots = list(envs.keys())
        self.device = next(iter(envs.values())).device

        # Universal action dim: policy outputs MAX_JOINTS actions for all robots.
        # Each robot's env receives only its actual DOF slice.
        self.num_actions = MAX_JOINTS

        self.num_envs = sum(e.num_envs for e in envs.values())
        self.max_episode_length = max(e.max_episode_length for e in envs.values())

        # Pre-compute per-robot row slices and actual action dims
        self._slices: dict[str, slice] = {}
        self._act_dims: dict[str, int] = {}
        offset = 0
        for name, env in envs.items():
            self._slices[name] = slice(offset, offset + env.num_envs)
            self._act_dims[name] = env.num_actions
            offset += env.num_envs

        n_act_str = ", ".join(f"{r}:{d}" for r, d in self._act_dims.items())
        print(
            f"[MultiRobotVecEnv] robots={self.robots}  "
            f"total_envs={self.num_envs}  "
            f"policy_num_actions={self.num_actions}  "
            f"per_robot_act_dims=({n_act_str})"
        )

    # ── rsl_rl VecEnv interface ────────────────────────────────────────────────

    @property
    def cfg(self):
        """Expose first robot's cfg (used only by wandb/neptune loggers)."""
        return next(iter(self.envs.values())).cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.cat([e.episode_length_buf for e in self.envs.values()])

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        for name, env in self.envs.items():
            env.episode_length_buf = value[self._slices[name]]

    def get_observations(self) -> TensorDict:
        """Concatenate observations from all robot envs."""
        parts = [env.get_observations()["policy"] for env in self.envs.values()]
        return TensorDict({"policy": torch.cat(parts, dim=0)}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step each robot env with its DOF-slice of the action tensor.

        Args:
            actions: (total_envs, MAX_JOINTS) — policy output on universal action space.

        Returns:
            obs_td : TensorDict{"policy": (total_envs, 78)}
            rewards: (total_envs,)
            dones  : (total_envs,)
            extras : {"log": {"{robot}/{key}": value, ...}}
        """
        obs_parts, rew_parts, done_parts = [], [], []
        log_merged: dict = {}

        for name, env in self.envs.items():
            sl = self._slices[name]
            robot_actions = actions[sl, : self._act_dims[name]]

            obs_td, rew, dones, extras = env.step(robot_actions)
            obs_parts.append(obs_td["policy"])
            rew_parts.append(rew)
            done_parts.append(dones)

            for k, v in extras.get("log", {}).items():
                log_merged[f"{name}/{k}"] = v

        obs_combined = TensorDict(
            {"policy": torch.cat(obs_parts, dim=0)},
            batch_size=[self.num_envs],
        )
        return (
            obs_combined,
            torch.cat(rew_parts, dim=0),
            torch.cat(done_parts, dim=0),
            {"log": log_merged},
        )

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all robot envs (called rarely; Isaac Lab auto-resets per episode)."""
        parts = []
        for env in self.envs.values():
            obs_td, _ = env.reset()
            parts.append(obs_td["policy"])
        return TensorDict({"policy": torch.cat(parts, dim=0)}, batch_size=[self.num_envs]), {}

    def close(self):
        for env in self.envs.values():
            env.close()
