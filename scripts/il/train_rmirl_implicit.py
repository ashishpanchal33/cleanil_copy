import torch
from cleanil.utils import (
    parse_configs, 
    set_seed, 
    get_device, 
    get_logger, 
    write_json,
    LoggerConfig,
)
from cleanil.data import (
    load_d4rl_expert_trajs, 
    load_d4rl_replay_buffer,
    combine_data, 
    normalize,
    train_test_split,
)
from cleanil.envs.utils import make_environment, EnvConfig
from cleanil.envs.termination import get_termination_fn
from cleanil.il import rmirl_implicit
from cleanil.rl.actor import make_tanh_normal_actor
from cleanil.rl.critic import DoubleQNetwork
from cleanil.dynamics.ensemble_dynamics import (
    EnsembleDynamics, 
    EnsembleConfig, 
    remove_non_topk_members,
    remove_reward_head,
)
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data import SamplerWithoutReplacement
from torchrl.envs.transforms import ObservationNorm

def main():
    config = parse_configs(
        EnvConfig(), 
        rmirl_implicit.RMIRLImplicitConfig(),
        LoggerConfig(),
        EnsembleConfig(),
    )
    env_config = EnvConfig(**config["env"])
    algo_config = rmirl_implicit.RMIRLImplicitConfig(**config["algo"])
    dynamics_config = EnsembleConfig(**config["dynamics"])
    write_json(config, f"{algo_config.save_path}/config.json")

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    print("device", device)

    # load pretrain dynamics
    dynamics_state_dict = torch.load(algo_config.pretrained_model_path, map_location=device)
    obs_mean = dynamics_state_dict["obs_mean"].to(device)
    obs_std = dynamics_state_dict["obs_std"].to(device)
    
    # load expert data
    expert_data = load_d4rl_expert_trajs(
        algo_config.expert_dataset, 
        algo_config.num_expert_trajs,
    )
    expert_data = expert_data.to(device)
    expert_data, eval_data = train_test_split(expert_data, algo_config.train_ratio)

    # load transition data
    buffer = load_d4rl_replay_buffer(
        algo_config.transition_dataset, 
        algo_config.batch_size,
    )
    buffer.set_sampler(SamplerWithoutReplacement())
    data = buffer.sample(batch_size=algo_config.transition_data_size)
    data = data.to(device)

    print("transition data size", len(data))

    # combine data, maybe upsample expert data for policy learning
    data = combine_data(expert_data, data, algo_config.upsample)

    # normalize data
    data["observation"] = normalize(data["observation"], obs_mean, obs_std**2)
    data["next"]["observation"] = normalize(data["next"]["observation"], obs_mean, obs_std**2)
    expert_data["observation"] = normalize(expert_data["observation"], obs_mean, obs_std**2)
    expert_data["next"]["observation"] = normalize(expert_data["next"]["observation"], obs_mean, obs_std**2)
    eval_data["observation"] = normalize(eval_data["observation"], obs_mean, obs_std**2)
    eval_data["next"]["observation"] = normalize(eval_data["next"]["observation"], obs_mean, obs_std**2)

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
    out_dim = obs_dim
    actor = make_tanh_normal_actor(
        obs_dim, 
        act_dim, 
        algo_config.hidden_dims, 
        algo_config.activation,
        action_spec.space.low,
        action_spec.space.high,
    )
    critic = DoubleQNetwork(
        obs_dim + 1 * algo_config.use_done, 
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
    state_dict = remove_reward_head(dynamics_state_dict["dynamics"], obs_dim)
    dynamics.load_state_dict(state_dict)
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
            len(data), 
            device=device,
        )
    )
    transition_buffer.extend(data)
    expert_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            len(expert_data), 
            device=device,
        )
    )
    expert_buffer.extend(expert_data)
    eval_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            len(eval_data), 
            device=device,
        )
    )
    eval_buffer.extend(eval_data)

    # setup trainer
    trainer = rmirl_implicit.Trainer(
        algo_config,
        actor,
        critic,
        dynamics,
        expert_buffer,
        transition_buffer,
        eval_buffer,
        eval_env,
        replay_buffer,
        termination_fn,
        logger,
        device,
    )
    trainer.train()

if __name__ == "__main__":
    main()