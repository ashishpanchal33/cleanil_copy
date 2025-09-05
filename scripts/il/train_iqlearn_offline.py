from cleanil.utils import (
    load_yaml, 
    set_seed, 
    get_device, 
    get_logger, 
    write_json,
    LoggerConfig,
)
from cleanil.data import normalize, train_test_split, load_custom_expert_trajs  # Remove load_d4rl_expert_trajs
from cleanil.il import iqlearn
from cleanil.rl.actor import make_tanh_normal_actor
from cleanil.rl.critic import DoubleQNetwork
from torchrl.data import LazyTensorStorage, ReplayBuffer


import torch

def main(data):
    config = load_yaml()
    algo_config = iqlearn.IQLearnConfig(**config["algo"])
    write_json(config, f"{algo_config.save_path}/config.json")

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    print("device", device)

    # CHANGE 1: Define your data dimensions manually (no environment needed)
    obs_dim = 4  # CartPole observation dimension
    act_dim = 1  # CartPole action dimension (discrete -> continuous conversion needed)
    action_bounds = (-1.0, 1.0)  # Set appropriate bounds

    # CHANGE 2: Load your custom data instead of D4RL
    expert_data = load_custom_expert_trajs(
        data, #"NathanGavenski/CartPole-v1",  # Your dataset name
        algo_config.num_expert_trajs,
        obs_dim,
        act_dim
    )
    
    expert_data = expert_data.to(device)
    expert_data, eval_data = train_test_split(expert_data, algo_config.train_ratio)

    # Normalize data
    obs_mean = expert_data["observation"].mean(0)
    obs_std = expert_data["observation"].std(0)
    expert_data["observation"] = normalize(expert_data["observation"], obs_mean, obs_std**2)
    expert_data["next"]["observation"] = normalize(expert_data["next"]["observation"], obs_mean, obs_std**2)
    eval_data["observation"] = normalize(eval_data["observation"], obs_mean, obs_std**2)
    eval_data["next"]["observation"] = normalize(eval_data["next"]["observation"], obs_mean, obs_std**2)
    
    # CHANGE 3: Create agent without environment specs
    actor = make_tanh_normal_actor(
        obs_dim, 
        act_dim, 
        algo_config.hidden_dims, 
        algo_config.activation,
        torch.tensor([action_bounds[0]] * act_dim),  # Manual action bounds
        torch.tensor([action_bounds[1]] * act_dim),
    )
    critic = DoubleQNetwork(
        obs_dim, 
        act_dim, 
        algo_config.hidden_dims, 
        algo_config.activation,
    )
    actor.to(device)
    critic.to(device)

    # Make buffers
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

    # CHANGE 4: Create trainer without eval_env
    trainer = iqlearn.OfflineTrainer(  # Use modified trainer
        algo_config,
        actor,
        critic,
        expert_buffer,
        eval_buffer,
        obs_mean,
        obs_std,
        act_dim,  # Pass action dimension instead of env
        logger,
        device,
    )
    trainer.train()

if __name__ == "__main__":
    main()
