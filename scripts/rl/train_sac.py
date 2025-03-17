from cleanil.utils import (
    load_yaml, 
    set_seed, 
    get_device, 
    get_logger, 
    LoggerConfig,
)
from cleanil.envs.utils import make_environment, EnvConfig
from cleanil.rl import sac
from cleanil.rl.actor import make_tanh_normal_actor
from cleanil.rl.critic import DoubleQNetwork
from torchrl.data import LazyTensorStorage, ReplayBuffer

def main():
    config = load_yaml()
    env_config = EnvConfig(**config["env"])
    algo_config = sac.SACConfig(**config["algo"])

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    print("device", device)
    
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
    actor.to(device)
    critic.to(device)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            algo_config.buffer_size, 
            device=device,
        )
    )

    # setup trainer
    trainer = sac.Trainer(
        algo_config,
        actor,
        critic,
        train_env,
        eval_env,
        replay_buffer,
        logger,
        device,
    )
    trainer.train()

if __name__ == "__main__":
    main()