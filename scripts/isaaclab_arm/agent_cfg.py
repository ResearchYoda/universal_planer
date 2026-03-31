"""rsl_rl runner configs for the universal pGraph arm policy."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class _PGraphActorCriticCfg(RslRlPpoActorCriticCfg):
    """pGraph Transformer policy — overrides class_name and removes MLP fields."""

    class_name: str = "PGraphTransformerActorCritic"
    init_noise_std: float = 1.0
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    # MLP fields are unused but required by the dataclass; set to empty lists
    actor_hidden_dims: list = []
    critic_hidden_dims: list = []
    activation: str = "elu"
    # Transformer hyper-parameters (passed as **kwargs to the policy)
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2


@configclass
class UniversalArmPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """OnPolicyRunner config for the pGraph arm reach task."""

    # Observation groups: both actor and critic see the "policy" obs group
    obs_groups: dict = {"policy": ["policy"], "critic": ["policy"]}

    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 100
    experiment_name: str = "pgraph_arm_reach"
    run_name: str = ""

    policy: _PGraphActorCriticCfg = _PGraphActorCriticCfg()
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
