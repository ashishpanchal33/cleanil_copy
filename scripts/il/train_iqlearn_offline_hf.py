
from cleanil.utils import (
    load_yaml, 
    set_seed, 
    get_device, 
    get_logger, 
    write_json,
    LoggerConfig,
)
from cleanil.data import normalize, train_test_split  # Remove load_d4rl_expert_trajs
from cleanil.il import iqlearn
from cleanil.rl.actor import make_tanh_normal_actor
from cleanil.rl.critic import DoubleQNetwork
from torchrl.data import LazyTensorStorage, ReplayBuffer

# Add your custom data loader
from datasets import load_dataset
import torch


import os


from datasets import load_dataset
import torch
from tensordict import TensorDict




def hf_to_tensordict(your_dataset_name,split: str = "train"):
    # 1. Load the dataset split
    ds = load_dataset(your_dataset_name, split=split)
    
    # 2. Extract columns
    obs = torch.tensor(ds["obs"], dtype=torch.float32)            # [T, obs_dim]

    
    actions = torch.tensor(ds["actions"], dtype=torch.float32)   # [T, act_dim]

    
    
    rewards = torch.tensor(ds["rewards"], dtype=torch.float32).unsqueeze(-1)  # [T,1]
    starts = torch.tensor(ds["episode_starts"], dtype=torch.bool) # [T]


    # FIX: Ensure actions are 2D
    if actions.ndim == 1:
        actions = actions.unsqueeze(-1)  # Convert [T] to [T, 1]
    
    T, obs_dim = obs.shape
    #act_dim = actions.shape[1] if actions.ndim > 1 else 1
    
    # 3. Build “terminated” and “truncated” masks
    #    episode_starts[i] == True means a new episode begins at i
    terminated = starts.roll(-1, dims=0)                          # shift left
    terminated[-1] = True                                          # last row ends episode
    truncated = torch.zeros_like(terminated)                       # no time limits
    
    # 4. Build “done” mask: episode ends either by termination or truncation
    done = terminated | truncated
    
    # 5. Build “next” observations and fields
    next_obs = obs.roll(-1, dims=0)
    next_obs[-1] = obs[-1]      # copy last frame for final step
    
    next_terminated = terminated.clone()
    next_truncated = truncated.clone()
    next_done = done.clone()
    
    # 6. Pack into a single TensorDict
    td = TensorDict(
        {
            "observation": obs,
            "action": actions,
            "reward": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
            "next": TensorDict(
                {
                    "observation": next_obs,
                    "reward": rewards,          # IQ-Learn ignores reward values
                    "terminated": next_terminated,
                    "truncated": next_truncated,
                    "done": next_done,
                },
                batch_size=[T],
            ),
        },
        batch_size=[T],
    )
    return td


# Then wrap in replay buffers and train as before.


def main():
    config = load_yaml()
    algo_config = iqlearn.IQLearnConfig(**config["algo"])

    # … after loading config and algo_config …
    os.makedirs(algo_config.save_path, exist_ok=True)
    write_json(config, f"{algo_config.save_path}/config.json")
    


    
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
    #expert_data = hf_to_tensordict(
    #    "NathanGavenski/CartPole-v1",  # Your dataset name
    #)


    # Example usage:
    expert_data = hf_to_tensordict("NathanGavenski/CartPole-v1","train")
    expert_data=expert_data.to(device)
    # Now you can split into train/test and feed into IQ-Learn buffers:
    from cleanil.data import train_test_split, normalize
    expert_data, eval_data = train_test_split(expert_data, train_ratio= algo_config.train_ratio)
    
    # Compute means/stds and normalize:
    obs_mean = expert_data["observation"].mean(0)
    obs_std = expert_data["observation"].std(0)
    expert_data["observation"] = normalize(expert_data["observation"], obs_mean, obs_std**2)
    expert_data["next"]["observation"] = normalize(expert_data["next"]["observation"], obs_mean, obs_std**2)
    eval_data["observation"] = normalize(eval_data["observation"], obs_mean, obs_std**2)
    eval_data["next"]["observation"] = normalize(eval_data["next"]["observation"], obs_mean, obs_std**2)
    
    
    
    









    
    
    #expert_data = expert_data.to(device)
    #expert_data, eval_data = train_test_split(expert_data, algo_config.train_ratio)

    # Normalize data
    #obs_mean = expert_data["observation"].mean(0)
    #obs_std = expert_data["observation"].std(0)
    #expert_data["observation"] = normalize(expert_data["observation"], obs_mean, obs_std**2)
    #expert_data["next"]["observation"] = normalize(expert_data["next"]["observation"], obs_mean, obs_std**2)
    #eval_data["observation"] = normalize(eval_data["observation"], obs_mean, obs_std**2)
    #eval_data["next"]["observation"] = normalize(eval_data["next"]["observation"], obs_mean, obs_std**2)
    
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
