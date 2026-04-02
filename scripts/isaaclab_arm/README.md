# pGraph Universal Arm Reach Policy — Isaac Lab

Single policy that controls **Franka Panda (7-DOF)**, **UR10 (6-DOF)**, and **Kinova Gen3 (7-DOF)** for end-effector reach tasks in Isaac Sim, using pGraph morphology encoding and a Transformer actor-critic.

---

## Core Idea

Each robot's kinematic chain is encoded as a graph where each joint is a node with 5 topology features. All robots are padded to `MAX_JOINTS=8`, giving a **fixed 78-dim observation** regardless of morphology. The same policy weights are used for all robots — only the pGraph feature slice (first 40 dims of the obs) differs.

```
pGraph node features per joint:
  [0] depth_norm   — position in chain (0 = base, 1 = EE-side)
  [1] joint_type   — 0=revolute, 1=prismatic
  [2] lim_lo_norm  — lower limit / π
  [3] lim_hi_norm  — upper limit / π
  [4] mask         — 1 if real joint, 0 if padding
```

---

## Architecture

```
Observation (78-dim, fixed for all robots)
─────────────────────────────────────────────
 [0:40]   pGraph features     8 joints × 5 dims  (static, robot topology)
 [40:48]  padded_joint_pos    8 dims              (zero-padded)
 [48:56]  padded_joint_vel    8 dims              (zero-padded)
 [56:63]  ee_pose             7 dims              [pos(3) + quat_wxyz(4)]
 [63:70]  goal_pose           7 dims              [pos(3) + quat_wxyz(4)]
 [70:78]  padded_last_action  8 dims              (zero-padded)

Token construction (10 tokens × 8 raw dims):
  Joint token i  : cat(pgraph[i]:5, pos[i]:1, vel[i]:1, act[i]:1)
  EE token    8  : cat(ee_pose:7, 0:1)
  Goal token  9  : cat(goal_pose:7, 0:1)

PGraphTransformerActorCritic
  token_proj   : Linear(8, 128)
  transformer  : TransformerEncoder(d_model=128, nhead=4, layers=2, pre-LN)
                 → src_key_padding_mask on pgraph[i,4]==0
  actor_head   : Linear(128, 1) applied to joint tokens [:num_actions]
  critic_head  : mean-pool valid tokens → Linear(128,128) → ELU → Linear(128,1)
  _log_std     : learnable (MAX_JOINTS,) parameter
```

**Universal action space:** `num_actions = MAX_JOINTS = 8`. The `_UniversalEnvWrapper` slices `actions[:, :robot_dof]` before passing to the environment, so the policy never changes shape between robots.

---

## Training

Isaac Lab allows only one `SimulationContext` per Python process, so multi-robot training is done via **curriculum fine-tuning** — alternate robots across separate runs:

```bash
# Step 1 — Train Franka from scratch
conda run -n env_isaaclab python scripts/isaaclab_arm/train.py \
    --robot franka --num_envs 1024 --iterations 1500 --headless

# Step 2 — Fine-tune on UR10 (loads Franka weights)
conda run -n env_isaaclab python scripts/isaaclab_arm/train.py \
    --robot ur10 --num_envs 1024 --iterations 1500 --headless \
    --resume logs/pgraph_arm/franka/<run>/model_1499.pt

# Step 3 — Fine-tune back on Franka (loads UR10 weights)
conda run -n env_isaaclab python scripts/isaaclab_arm/train.py \
    --robot franka --num_envs 1024 --iterations 1500 --headless \
    --resume logs/pgraph_arm/ur10/<run>/model_1499.pt

# Repeat alternating for more rounds
```

Checkpoints are saved every 100 iterations to `logs/pgraph_arm/<robot>/<timestamp>/`.

---

## Evaluation

```bash
# Headless eval (prints pos/ori error over N episodes)
conda run -n env_isaaclab python scripts/isaaclab_arm/test.py \
    --robot ur10 \
    --checkpoint logs/pgraph_arm/ur10/<run>/model_900.pt \
    --num_envs 4 --episodes 50 --headless

# Live Isaac Sim window
conda run -n env_isaaclab python scripts/isaaclab_arm/test.py \
    --robot franka \
    --checkpoint logs/pgraph_arm/franka/<run>/model_1499.pt \
    --num_envs 16 --episodes 200
```

The script automatically resizes `_log_std` if the checkpoint was trained on a different DOF robot.

---

## Curriculum Training Results

| Round | Robot | Direction | Pos Error (train) | Pos Error (demo 50ep) | Ori Error (demo) |
|---|---|---|---|---|---|
| R1 | Franka | scratch | 0.033 m | **0.033 m** | — |
| R2 | UR10 | ← Franka R1 | 0.340 m | 0.310 m | 30.9° |
| R3 | Franka | ← UR10 R2 | 0.089 m | 0.098 m | 19.3° |
| R4 | UR10 | ← Franka R3 | 0.221 m | **0.174 m** | 40.0° |

UR10 improves ~44% per curriculum cycle (0.340 → 0.174 m after 2 rounds). Franka remains stable. More rounds expected to push UR10 below 0.1 m.

### Why UR10 lags behind Franka

1. **Joint limits are 2× wider** — UR10 ±6.28 rad vs Franka ±2.9 rad. The policy must learn a larger action space.
2. **Transfer direction matters** — policy pre-trained on Franka's tighter workspace needs to unlearn some habits.
3. **More rounds needed** — literature suggests 3–5 curriculum cycles for stable cross-morphology convergence.

---

## Robot Configs

| Robot | DOF | EE Body | Joint Pattern | Checkpoint |
|---|---|---|---|---|
| Franka Panda | 7 | `panda_hand` | `panda_joint.*` | `franka/2026-03-31.../model_1499.pt` |
| UR10 | 6 | `ee_link` | `.*` | `ur10/2026-04-02.../model_900.pt` |
| Kinova Gen3 | 7 | `end_effector_link` | `joint_[1-7]` | — (not yet trained) |

---

## File Overview

| File | Role |
|---|---|
| `policy.py` | `PGraphTransformerActorCritic` — rsl_rl compatible actor-critic |
| `pgraph.py` | Per-robot topology config + `compute_pgraph_features()` |
| `env_cfg.py` | `UniversalArmReachEnvCfg` — 78-dim fixed obs, inherits Isaac Lab ReachEnvCfg |
| `agent_cfg.py` | `UniversalArmPPORunnerCfg` — PPO hyperparameters |
| `train.py` | Training script with `--resume` curriculum support + `_UniversalEnvWrapper` |
| `test.py` | Evaluation loop — headless or live Isaac Sim window |
| `multi_robot_env.py` | `MultiRobotVecEnv` — designed for joint training; currently blocked by Isaac Lab's single-SimulationContext constraint |
| `mdp/observations.py` | Custom obs terms: `pgraph_features`, `padded_joint_pos/vel`, `ee_pose_in_env` |
| `configs/franka_cfg.py` | Franka train + play env configs |
| `configs/ur10_cfg.py` | UR10 train + play env configs |
| `configs/kinova_cfg.py` | Kinova Gen3 train + play env configs |

---

## PPO Hyperparameters

| Parameter | Value |
|---|---|
| `num_steps_per_env` | 24 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `learning_rate` | 3e-4 (adaptive) |
| `clip_param` | 0.2 |
| `entropy_coef` | 0.005 |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `desired_kl` | 0.01 |

---

## Known Limitations & TODO

- [ ] Isaac Lab single-SimulationContext prevents true joint multi-robot training — `MultiRobotVecEnv` exists but cannot be used in practice
- [ ] Kinova Gen3 not yet trained
- [ ] UR10 orientation error still high (40°) — needs more curriculum rounds
- [ ] Add per-robot action scaling based on joint range widths
- [ ] Evaluate zero-shot: Franka policy on Kinova (both 7-DOF, similar topology)
- [ ] Add link-length features to pGraph for finer morphology discrimination
