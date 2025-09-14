from cleanil.utils import (
    load_yaml, set_seed, get_device, get_logger, 
    write_json, LoggerConfig,
)
from cleanil.data import normalize, train_test_split
from cleanil.il.iqlearn_discrete import DiscreteOfflineTrainer, IQLearnConfig
#from cleanil.il.iqlearn import IQLearnConfig
#from cleanil.rl.actor_discrete import make_categorical_actor
from cleanil.rl.critic_discrete import DoubleQNetwork
from cleanil.rl.critic_discrete_LSTM import HistoryDoubleQNetwork
from cleanil.rl.critic_discrete_transformer import HistoryDoubleQNetwork_transformer



from torchrl.data import LazyTensorStorage, ReplayBuffer
import os
from datasets import load_dataset
import torch
from tensordict import TensorDict
import pandas as pd


import torch
from collections import deque
import numpy as np



def hf_to_tensordict_discrete(dataset_name, split="train", load_local_csv = False, id_=None):
    
    if load_local_csv:
        ds = pd.read_pickle(dataset_name) # in this case dataset_name is datapath
    else:
        
        ds = load_dataset(dataset_name, split=split).to_pandas()
    
    obs = torch.tensor(ds["obs"], dtype=torch.float32)
    # Keep actions as discrete integers but add dimension for compatibility
    actions = torch.tensor(ds["actions"], dtype=torch.long)#.unsqueeze(-1)
    rewards = torch.tensor(ds["rewards"], dtype=torch.float32).unsqueeze(-1)
    starts = torch.tensor(ds["episode_starts"], dtype=torch.bool)

    print(id_)

    if id_ == None:
        ds['id']  = ds.episode_starts.cumsum()
    else:
        ds['id'] = ds[id_]

    def get_next_obs(x):
        x = x.copy()
    
        x['next_obs'] = x['obs'].shift(-1).values
        las_val = x.loc[x.index[-1],'obs']
        x.at[x.index[-1], 'next_obs'] = las_val
    
        
        return x

    ds = ds.groupby('id').apply(get_next_obs).reset_index(drop=True)

    
    next_obs = torch.tensor(ds["next_obs"], dtype=torch.float32)

    
    #actions = torch.tensor(ds["actions"], dtype=torch.float32)   # [T, act_dim]
    #rewards = torch.tensor(ds["rewards"], dtype=torch.float32).unsqueeze(-1)  # [T,1]
    #starts = torch.tensor(ds["episode_starts"], dtype=torch.bool) # [T]


    
    
    T = obs.shape[0]
    terminated = starts.roll(-1, dims=0)
    terminated[-1] = True
    truncated = torch.zeros_like(terminated)
    done = terminated | truncated
    
    
    
    td = TensorDict({
        "observation": obs,
        "action": actions,
        "reward": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
        "next": TensorDict({
            "observation": next_obs,
            "reward": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
        }, batch_size=[T]),
    }, batch_size=[T])
    
    return td








def create_history_stacked_dataset(data: TensorDict, sequence_length: int):
    """
    Transforms a dataset by stacking a history of observations for each time step.
    Handles episode boundaries by padding with zeros.

    Args:
        data (TensorDict): The original dataset.
        sequence_length (int): The number of historical frames to stack.

    Returns:
        TensorDict: A new dataset with history-stacked observations.
    """
    original_obs = data["observation"]
    episode_starts = data["terminated"] # Using terminated as the start of the *next* episode
    
    num_timesteps, obs_dim = original_obs.shape
    
    # The new observation will be a flat vector of the stacked history
    new_obs = torch.zeros(num_timesteps, sequence_length * obs_dim, dtype=torch.float32)
    
    history = deque(maxlen=sequence_length)
    
    print("Creating history-stacked observations...")
    for t in range(num_timesteps):
        # If it's the start of a new episode, clear the history and pad with zeros
        if episode_starts[t-1] and t > 0: # Check the *previous* step's done flag
            history.clear()
        
        # Add the current observation to the history
        history.append(original_obs[t])
        
        # Create the padded history
        padded_history = list(history)
        while len(padded_history) < sequence_length:
            # Pad at the beginning (older timesteps) with zeros
            padded_history.insert(0, torch.zeros(obs_dim, dtype=torch.float32))
            
        # Flatten and store the result
        new_obs[t] = torch.cat(padded_history).flatten()

    # Create a new TensorDict with the modified observations
    # Note: next_obs also needs to be transformed!
    history_data = data.clone()
    history_data["observation"] = new_obs
    
    # We also need to create the history for the 'next_observation'
    # The easiest way is to apply the same logic to the next_obs tensor.
    # A quick way is to roll the whole dataset by -1.
    next_obs_history = torch.roll(new_obs, shifts=-1, dims=0)
    # At the very end of the dataset, the next_obs is itself.
    next_obs_history[-1] = new_obs[-1]
    # At episode boundaries, the next_obs is the start of a new history.
    done_indices = data["done"].nonzero(as_tuple=True)[0]
    for idx in done_indices:
        if idx + 1 < num_timesteps:
            next_obs_history[idx] = new_obs[idx+1]
            
    history_data["next"]["observation"] = next_obs_history
    
    print("History stacking complete.")
    return history_data























def main():
    config = load_yaml()
    algo_config = IQLearnConfig(**config["algo"])
    
    os.makedirs(algo_config.save_path, exist_ok=True)
    write_json(config, f"{algo_config.save_path}/config.json")
    
    set_seed(config["seed"])
    device = get_device(config["device"])
    logger = get_logger(config, LoggerConfig(**config["logger"]))
    
    # Define dimensions for discrete CartPole
    original_obs_dim = algo_config.obs_dim #4
    act_dim = algo_config.act_dim #2  # Binary actions: 0 or 1
    num_actions  = algo_config.num_actions


    #act_dim : 1
    
    #num_actions: 2
    # Load discrete data
    whole_data_ = hf_to_tensordict_discrete(algo_config.dataset, "train",load_local_csv = algo_config.load_local_csv,id_=algo_config.id_)
    
    if algo_config.sequential:
        SEQUENCE_LENGTH = algo_config.SEQUENCE_LENGTH #4 # How many steps of history to use
        whole_data = create_history_stacked_dataset(whole_data_, SEQUENCE_LENGTH)
        whole_data = whole_data.to(device)
        obs_dim = SEQUENCE_LENGTH * original_obs_dim

    else:
        obs_dim = original_obs_dim
        whole_data = whole_data_.to(device)


    


    
    
    expert_data, eval_data = train_test_split(whole_data, algo_config.train_ratio)
    
    # Normalize observations
    obs_mean = expert_data["observation"].mean(0)
    obs_std = expert_data["observation"].std(0)
    expert_data["observation"] = normalize(expert_data["observation"], obs_mean, obs_std**2)
    expert_data["next"]["observation"] = normalize(expert_data["next"]["observation"], obs_mean, obs_std**2)
    eval_data["observation"] = normalize(eval_data["observation"], obs_mean, obs_std**2)
    eval_data["next"]["observation"] = normalize(eval_data["next"]["observation"], obs_mean, obs_std**2)

    
    whole_data["observation"] = normalize(whole_data["observation"], obs_mean, obs_std**2)
    whole_data["next"]["observation"] = normalize(whole_data["next"]["observation"], obs_mean, obs_std**2)


    




    if algo_config.model == 'lstm':

        critic = HistoryDoubleQNetwork(
            obs_dim, 
            num_actions, 
            algo_config.hidden_dims, 
            algo_config.activation, 
            sequence_length=SEQUENCE_LENGTH
        )

    elif algo_config.model == 'transformer':

        critic = HistoryDoubleQNetwork_transformer(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dims=algo_config.hidden_dims,
            activation=algo_config.activation,
            sequence_length=SEQUENCE_LENGTH,
            nhead=algo_config.nhead,  # Example hyperparameter for the transformer
            num_layers=algo_config.num_layers # Example hyperparameter for the transformer
        )


    
    else:

        # Create discrete critic
        #actor = make_categorical_actor(obs_dim, act_dim, algo_config.hidden_dims, algo_config.activation)
        #obs_dim: int, num_actions: int, hidden_dims: list, activation: str
        critic = DoubleQNetwork(obs_dim, num_actions, algo_config.hidden_dims, algo_config.activation, dropout = algo_config.dropout)  # 1D action for critic
        
    #actor.to(device)
    critic.to(device)
    
    # Create buffers
    expert_buffer = ReplayBuffer(storage=LazyTensorStorage(len(expert_data), device=device))
    expert_buffer.extend(expert_data)
    
    eval_buffer = ReplayBuffer(storage=LazyTensorStorage(len(eval_data), device=device))
    eval_buffer.extend(eval_data)


    whole_data_buffer = ReplayBuffer(storage=LazyTensorStorage(len(whole_data), device=device))
    whole_data_buffer.extend(whole_data)
    
    # Create discrete trainer
    trainer = DiscreteOfflineTrainer(
        algo_config, critic, expert_buffer, eval_buffer,
        obs_mean, obs_std, 1, logger, device ,whole_data_buffer = whole_data_buffer # act_dim=1 for critic compatibility
    )
    trainer.train()

if __name__ == "__main__":
    main()
