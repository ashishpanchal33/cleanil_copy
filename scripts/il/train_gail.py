from cleanil.utils import (
    load_yaml, 
    set_seed, 
    get_device, 
    get_logger, 
    write_json,
    LoggerConfig,
)
from cleanil.data import load_d4rl_expert_trajs
from cleanil.envs.utils import make_environment, EnvConfig
from cleanil.il import gail
from cleanil.il.reward import RewardModel
from cleanil.rl.actor import make_tanh_normal_actor
from cleanil.rl.critic import DoubleQNetwork
from torchrl.data import LazyTensorStorage, ReplayBuffer

def main():
    config = load_yaml()
    env_config = EnvConfig(**config["env"])
    algo_config = gail.GAILConfig(**config["algo"])
    write_json(config, f"{algo_config.save_path}/config.json")

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    print("device", device)

    # load expert data
    expert_data = load_d4rl_expert_trajs(
        algo_config.dataset, 
        algo_config.num_expert_trajs,
    )
    expert_data = expert_data.to(device)
    
    # setup environments
    train_env, eval_env = make_environment(env_config, device)

    # setup agent
    observation_spec = train_env.observation_spec["observation"]
    action_spec = train_env.action_spec
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
    critic = DoubleQNetwork(
        obs_dim, 
        act_dim, 
        algo_config.hidden_dims, 
        algo_config.activation,
    )
    reward = RewardModel(
        obs_dim, 
        act_dim, 
        algo_config.hidden_dims, 
        algo_config.activation,
        algo_config.reward_state_only,
        algo_config.reward_use_done,
        algo_config.max_reward,
    )
    actor.to(device)
    critic.to(device)
    reward.to(device)

    # make buffers
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            algo_config.buffer_size, 
            device=device,
        )
    )
    expert_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            len(expert_data), 
            device=device,
        )
    )
    expert_buffer.extend(expert_data)

    # setup trainer
    trainer = gail.Trainer(
        algo_config,
        actor,
        critic,
        reward,
        expert_buffer,
        train_env,
        eval_env,
        replay_buffer,
        logger,
        device,
    )
    trainer.train()

if __name__ == "__main__":
    main()