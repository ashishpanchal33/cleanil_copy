from cleanil.utils import (
    load_yaml, 
    set_seed, 
    get_device, 
    get_logger, 
    LoggerConfig,
)
from cleanil.envs.utils import make_environment, EnvConfig
from cleanil.envs.termination import get_termination_fn
from cleanil.rl import mbpo
from cleanil.rl.actor import make_tanh_normal_actor
from cleanil.rl.critic import DoubleQNetwork
from cleanil.dynamics.ensemble_dynamics import EnsembleDynamics, EnsembleConfig
from torchrl.data import LazyTensorStorage, ReplayBuffer

def main():
    config = load_yaml()
    env_config = EnvConfig(**config["env"])
    algo_config = mbpo.MBPOConfig(**config["algo"])
    dynamics_config = EnsembleConfig(**config["dynamics"])

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    print("device", device)
    
    # setup environments
    train_env, eval_env = make_environment(env_config, device)
    termination_fn = get_termination_fn(env_config.name)

    # setup agent
    observation_spec = train_env.observation_spec["observation"]
    action_spec = train_env.action_spec
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

    # make buffers
    rollout_batch_size = algo_config.rollout_batch_size
    rollout_max_steps = algo_config.rollout_max_steps
    model_retain_epochs = algo_config.model_retain_epochs
    sample_model_count = algo_config.steps_per_epoch // algo_config.update_model_every
    model_steps = int(rollout_batch_size * rollout_max_steps * sample_model_count * model_retain_epochs)
    buffer_size = min(algo_config.buffer_size, model_steps)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            buffer_size, 
            device=device,
        )
    )

    # setup trainer
    trainer = mbpo.Trainer(
        algo_config,
        actor,
        critic,
        dynamics,
        train_env,
        eval_env,
        replay_buffer,
        termination_fn,
        logger,
        device,
    )
    trainer.train()

if __name__ == "__main__":
    main()