from cleanil.utils import (
    parse_configs, 
    set_seed, 
    get_device, 
    get_logger, 
    write_json,
    LoggerConfig,
)
from cleanil.data import load_d4rl_expert_trajs, normalize, train_test_split
from cleanil.envs.utils import make_environment, EnvConfig
from cleanil.il import bc
from cleanil.rl.actor import make_tanh_normal_actor
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.transforms import ObservationNorm

def main():
    config = parse_configs(
        EnvConfig(), 
        bc.BCConfig(),
        LoggerConfig(),
    )
    env_config = EnvConfig(**config["env"])
    algo_config = bc.BCConfig(**config["algo"])
    write_json(config, f"{algo_config.save_path}/config.json")

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    print("device", device)

    # load expert data
    expert_data = load_d4rl_expert_trajs(
        algo_config.dataset,
        algo_config.num_expert_trajs,
        skip_terminated=True,
    )
    expert_data = expert_data.to(device)
    expert_data, eval_data = train_test_split(expert_data, algo_config.train_ratio)

    # normalize data
    obs_mean = expert_data["observation"].mean(0)
    obs_std = expert_data["observation"].std(0)
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

    # setup agent
    observation_spec = eval_env.observation_spec["observation"]
    action_spec = eval_env.action_spec
    obs_dim = observation_spec.shape[-1]
    act_dim = action_spec.shape[-1]
    actor = make_tanh_normal_actor(
        obs_dim, 
        act_dim, 
        algo_config.hidden_dims, 
        algo_config.activation,
        action_spec.space.low,
        action_spec.space.high,
    )
    actor.to(device)

    # make buffers
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
    trainer = bc.Trainer(
        algo_config,
        actor,
        expert_buffer,
        eval_buffer,
        obs_mean,
        obs_std,
        eval_env,
        logger,
        device,
    )
    trainer.train()

if __name__ == "__main__":
    main()