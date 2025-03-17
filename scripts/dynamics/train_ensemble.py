from dataclasses import dataclass
import torch
import torch.nn as nn
from cleanil.utils import (
    load_yaml, 
    set_seed, 
    get_device, 
    get_logger, 
    LoggerConfig,
)
from cleanil.base_trainer import BaseTrainerConfig, BaseTrainer
from cleanil.dynamics.ensemble_dynamics import (
    EnsembleConfig,
    EnsembleDynamics, 
    train_ensemble,
)
from cleanil.data import (
    load_d4rl_expert_trajs, 
    load_d4rl_replay_buffer, 
    combine_data,
    normalize,
)
from tensordict import TensorDict
from torchrl.data import SamplerWithoutReplacement


@dataclass
class TrainConfig(BaseTrainerConfig):
    # data args
    expert_dataset: str = "halfcheetah-expert-v2"
    transition_dataset: str = "halfcheetah-medium-replay-v2"
    num_expert_trajs: int = 10
    transition_data_size: int = 1000000
    upsample: bool = False

    # train args
    pred_reward: bool = True
    bootstrap: bool = True
    lr_model: float = 3e-4
    grad_clip: float | None = None
    model_eval_ratio: float = 0.2
    model_train_batch_size: int = 256
    model_train_steps: int = 500


class Trainer(BaseTrainer):
    def __init__(
        self, 
        config: TrainConfig, 
        data: TensorDict,
        dynamics: EnsembleDynamics,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.data = data
        self.dynamics = dynamics
        self.optimizer = torch.optim.Adam(dynamics.parameters(), lr=config.lr_model)

        self._modules["dynamics"] = self.dynamics
        self._modules["obs_mean"] = nn.Parameter(obs_mean, requires_grad=False)
        self._modules["obs_std"] = nn.Parameter(obs_std, requires_grad=False)

    def train(self):
        config = self.config
        data = self.data

        train_ensemble(
            data,
            config.pred_reward,
            self.dynamics,
            self.optimizer,
            config.model_eval_ratio,
            config.model_train_batch_size,
            config.model_train_steps,
            bootstrap=config.bootstrap,
            grad_clip=config.grad_clip,
            logger=self.logger
        )


def main():
    config = load_yaml()
    algo_config = TrainConfig(**config["algo"])
    dynamics_config = EnsembleConfig(**config["dynamics"])

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    print("device", device)

    # load expert data
    expert_data = load_d4rl_expert_trajs(
        algo_config.expert_dataset, 
        algo_config.num_expert_trajs,
    )

    # load transition data
    buffer = load_d4rl_replay_buffer(
        algo_config.transition_dataset, 
        algo_config.model_train_batch_size,
    )
    buffer.set_sampler(SamplerWithoutReplacement())
    data = buffer.sample(batch_size=algo_config.transition_data_size)

    print("transition data size", len(data))

    # combine data
    data = combine_data(expert_data, data, algo_config.upsample)
    data = data.to(device)

    # normalize data
    obs_mean = data["observation"].mean(0)
    obs_std = data["observation"].std(0)
    data["observation"] = normalize(data["observation"], obs_mean, obs_std**2)
    data["next"]["observation"] = normalize(data["next"]["observation"], obs_mean, obs_std**2)
    
    # setup dynamics
    obs_dim = data["observation"].shape[-1]
    act_dim = data["action"].shape[-1]
    out_dim = obs_dim + 1
    dynamics = EnsembleDynamics(
        obs_dim,
        act_dim,
        out_dim,
        dynamics_config,
    )
    dynamics.to(device)

    trainer = Trainer(
        algo_config,
        data,
        dynamics,
        obs_mean,
        obs_std,
        logger=logger,
    )
    trainer.train()
    trainer.save()

if __name__ == "__main__":
    main()