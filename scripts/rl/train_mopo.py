import torch
from cleanil.utils import (
    load_yaml, 
    set_seed, 
    get_device, 
    get_logger, 
    LoggerConfig,
)
from cleanil.data import load_d4rl_replay_buffer, normalize
from cleanil.envs.utils import make_environment, EnvConfig
from cleanil.envs.termination import get_termination_fn
from cleanil.rl import mopo
from cleanil.rl.actor import make_tanh_normal_actor
from cleanil.rl.critic import DoubleQNetwork
from cleanil.dynamics.ensemble_dynamics import EnsembleDynamics, EnsembleConfig
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.transforms import ObservationNorm

def main():
    config = load_yaml()
    env_config = EnvConfig(**config["env"])
    algo_config = mopo.MOPOConfig(**config["algo"])
    dynamics_config = EnsembleConfig(**config["dynamics"])

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    print("device", device)

    # load pretrain dynamics
    dynamics_state_dict = torch.load(algo_config.pretrained_model_path, map_location=device)
    obs_mean = dynamics_state_dict["obs_mean"].to(device)
    obs_std = dynamics_state_dict["obs_std"].to(device)
    
    # load offline data
    buffer = load_d4rl_replay_buffer(
        algo_config.dataset, 
        algo_config.batch_size,
    )
    data = buffer.sample(batch_size=algo_config.transition_data_size)
    data = data.to(device)

    # normalize data
    data["observation"] = normalize(data["observation"], obs_mean, obs_std**2)
    data["next"]["observation"] = normalize(data["next"]["observation"], obs_mean, obs_std**2)

    # setup environments
    _, eval_env = make_environment(env_config, device)
    transform = ObservationNorm(
        obs_mean, obs_std, 
        in_keys=["observation"], out_keys=["observation"],
        standard_normal=True,
    )
    eval_env.append_transform(transform)
    termination_fn = get_termination_fn(
        env_config.name,
        obs_mean,
        obs_std,
    )

    # setup agent
    observation_spec = eval_env.observation_spec["observation"]
    action_spec = eval_env.action_spec
    obs_dim = observation_spec.shape[-1]
    act_dim = action_spec.shape[-1]
    out_dim = obs_dim + 1
    actor = make_tanh_normal_actor(
        obs_dim, 
        act_dim, 
        algo_config.hidden_dims, 
        algo_config.activation,
        action_spec.space.low,
        action_spec.space.high,
    )
    critic = DoubleQNetwork(
        obs_dim, 
        act_dim, 
        algo_config.hidden_dims, 
        algo_config.activation,
    )
    dynamics = EnsembleDynamics(
        obs_dim,
        act_dim,
        out_dim,
        dynamics_config,
    )
    actor.to(device)
    critic.to(device)
    dynamics.to(device)

    # load pretrained dynamics
    dynamics.load_state_dict(dynamics_state_dict["dynamics"])

    # make buffers
    rollout_batch_size = algo_config.rollout_batch_size
    rollout_max_steps = algo_config.rollout_max_steps
    sample_model_every = algo_config.sample_model_every
    model_retain_epochs = algo_config.model_retain_epochs
    model_steps = int(rollout_batch_size * rollout_max_steps * sample_model_every * model_retain_epochs)
    buffer_size = min(algo_config.buffer_size, model_steps)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            buffer_size, 
            device=device,
        )
    )
    transition_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            algo_config.buffer_size, 
            device=device,
        )
    )
    transition_buffer.extend(data)

    # setup trainer
    trainer = mopo.Trainer(
        algo_config,
        actor,
        critic,
        dynamics,
        transition_buffer,
        eval_env,
        termination_fn,
        replay_buffer,
        logger,
        device,
    )
    trainer.train()

if __name__ == "__main__":
    main()