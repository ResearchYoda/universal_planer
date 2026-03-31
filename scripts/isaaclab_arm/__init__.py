"""Universal pGraph Arm Reach — Isaac Lab extension.

Gym environments registered here:
    Isaac-PGraph-Reach-Franka-v0
    Isaac-PGraph-Reach-Franka-Play-v0
    Isaac-PGraph-Reach-UR10-v0
    Isaac-PGraph-Reach-UR10-Play-v0
    Isaac-PGraph-Reach-KinovaGen3-v0
    Isaac-PGraph-Reach-KinovaGen3-Play-v0
"""

import gymnasium as gym

# NOTE: isaaclab imports (configclass, etc.) require AppLauncher to be started first
# (pxr/Isaac Sim must be loaded). This file only registers gym entry points (string refs).

# ── Franka Panda ──────────────────────────────────────────────────────────────
gym.register(
    id="Isaac-PGraph-Reach-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "scripts.isaaclab_arm.configs.franka_cfg:FrankaPGraphReachEnvCfg",
        "rsl_rl_cfg_entry_point": "scripts.isaaclab_arm.agent_cfg:UniversalArmPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PGraph-Reach-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "scripts.isaaclab_arm.configs.franka_cfg:FrankaPGraphReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "scripts.isaaclab_arm.agent_cfg:UniversalArmPPORunnerCfg",
    },
)

# ── UR10 ──────────────────────────────────────────────────────────────────────
gym.register(
    id="Isaac-PGraph-Reach-UR10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "scripts.isaaclab_arm.configs.ur10_cfg:UR10PGraphReachEnvCfg",
        "rsl_rl_cfg_entry_point": "scripts.isaaclab_arm.agent_cfg:UniversalArmPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PGraph-Reach-UR10-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "scripts.isaaclab_arm.configs.ur10_cfg:UR10PGraphReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "scripts.isaaclab_arm.agent_cfg:UniversalArmPPORunnerCfg",
    },
)

# ── Kinova Gen3 ───────────────────────────────────────────────────────────────
gym.register(
    id="Isaac-PGraph-Reach-KinovaGen3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "scripts.isaaclab_arm.configs.kinova_cfg:KinovaGen3PGraphReachEnvCfg",
        "rsl_rl_cfg_entry_point": "scripts.isaaclab_arm.agent_cfg:UniversalArmPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PGraph-Reach-KinovaGen3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "scripts.isaaclab_arm.configs.kinova_cfg:KinovaGen3PGraphReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "scripts.isaaclab_arm.agent_cfg:UniversalArmPPORunnerCfg",
    },
)
