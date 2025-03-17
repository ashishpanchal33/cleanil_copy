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
from cleanil.rl import rambo
from cleanil.rl.actor import make_tanh_normal_actor
from cleanil.rl.critic import DoubleQNetwork
from cleanil.dynamics.ensemble_dynamics import (
    EnsembleDynamics, 
    EnsembleConfig, 
    remove_non_topk_members,
)
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement, RandomSampler
from torchrl.envs.transforms import ObservationNorm

def main():
    config = load_yaml()
    env_config = EnvConfig(**config["env"])
    algo_config = rambo.RAMBOConfig(**config["algo"])
    dynamics_config = EnsembleConfig(**config["dynamics"])

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    print("device", device)

    # load pretrain dynamics
    dynamics_state_dict = torch.load(algo_config.pretrained_model_path, map_location=torch.device("cpu"))
    obs_mean = dynamics_state_dict["obs_mean"].to(device)
    obs_std = dynamics_state_dict["obs_std"].to(device)
    
    # load offline data
    buffer = load_d4rl_replay_buffer(
        algo_config.dataset, 
        algo_config.batch_size,
    )
    buffer.set_sampler(SamplerWithoutReplacement())
    data = buffer.sample(batch_size=algo_config.transition_data_size)
    buffer.set_sampler(RandomSampler())
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
    
    # load pretrained dynamics
    dynamics.load_state_dict(dynamics_state_dict["dynamics"])
    dynamics = remove_non_topk_members(dynamics, dynamics_config)

    actor.to(device)
    critic.to(device)
    dynamics.to(device)

    # make buffers
    rollout_batch_size = algo_config.rollout_batch_size
    rollout_max_steps = algo_config.rollout_max_steps
    model_retain_epochs = algo_config.model_retain_epochs
    sample_model_count = algo_config.steps_per_epoch // algo_config.sample_model_every
    model_steps = int(rollout_batch_size * rollout_max_steps * sample_model_count * model_retain_epochs)
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
    trainer = rambo.Trainer(
        algo_config,
        actor,
        critic,
        dynamics,
        transition_buffer,
        eval_env,
        replay_buffer,
        termination_fn,
        logger,
        device,
    )
    trainer.train()

if __name__ == "__main__":
    main()