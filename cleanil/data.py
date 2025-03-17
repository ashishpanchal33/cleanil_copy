import torch
from tqdm import tqdm
from cleanil.utils import concat_tensordict_on_shared_keys
from tensordict import TensorDict
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import TensorDictReplayBuffer, SamplerWithoutReplacement
from torchrl.data import LazyTensorStorage
from torchrl.envs import DoubleToFloat

def normalize(x, mean, variance):
    return (x - mean) / variance**0.5

def denormalize(x, mean, variance):
    return x * variance**0.5 + mean

def load_d4rl_replay_buffer(dataset: str, batch_size: int) -> TensorDictReplayBuffer:
    data = D4RLExperienceReplay(
        dataset_id=dataset,
        split_trajs=True,
        batch_size=batch_size,
        prefetch=4,
        direct_download=True,
        sampler=SamplerWithoutReplacement()
    )
    data.append_transform(DoubleToFloat())

    # handle shapes of different datasets
    data = data.storage._storage.float()
    
    # flatten non-expert trajectory data
    if len(data.shape) > 1:
        data = data.flatten()
        data = data[data["mask"] == 1]

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(len(data)),
        batch_size=batch_size,
    )
    buffer.extend(data)
    return buffer

def load_d4rl_expert_trajs(
    dataset: str, 
    num_expert_trajs: int, 
    skip_terminated: bool = True, 
) -> TensorDict:
    """Load d4rl dataset in trajectory form. Sampling full trajectories can be done using SliceSampler
    
    Args:
        dataset (str): dataset name
        num_expert_trajs (int): number of random expert trajectories to keep
        skip_terminated (bool): whether to include expert trajectories with terminated
    """
    data = load_d4rl_replay_buffer(dataset, 32)
    data = data._storage._storage

    eps_id = torch.stack([data["done"]]).any(0).cumsum(0).flatten()
    
    # shuffle unique eps id
    unique_eps_id = eps_id.unique()
    unique_eps_id = unique_eps_id[torch.randperm(len(unique_eps_id))]
    
    print("parsing d4rl stacked trajectories")
    traj_dataset = []
    for e in tqdm(torch.unique(eps_id)):
        if data["terminated"][eps_id == e].sum() > 0 and skip_terminated:
            continue

        traj_dataset.append(data[eps_id == e])

        if len(traj_dataset) >= num_expert_trajs:
            break

    traj_dataset = torch.cat(traj_dataset, dim=0)
    traj_dataset["reward"] *= 0.
    traj_dataset["next"]["reward"] *= 0.
    traj_dataset.del_("info")
    
    traj_dataset["truncated"] = traj_dataset["truncated"].bool()
    traj_dataset["next"]["truncated"] = traj_dataset["next"]["truncated"].bool()
    traj_dataset["terminated"] = traj_dataset["terminated"].bool()
    traj_dataset["next"]["terminated"] = traj_dataset["next"]["terminated"].bool()
    traj_dataset["done"] = traj_dataset["done"].bool()
    traj_dataset["next"]["done"] = traj_dataset["next"]["done"].bool()
    return traj_dataset

def combine_data(expert_data: TensorDict, transition_data: TensorDict, upsample: bool = True):
    _expert_data = expert_data
    if upsample:
        idx = torch.randint(0, len(_expert_data), (len(transition_data),))
        _expert_data = _expert_data[idx]
    
    transition_data = concat_tensordict_on_shared_keys(_expert_data, transition_data)
    return transition_data

def train_test_split(data: TensorDict, train_ratio: float, max_eval_size: float = 1000):
    """Split at random. This will shuffle data."""
    idx = torch.randperm(len(data))
    data = data[idx]
    eval_size = round(len(data) * (1 - train_ratio))
    eval_size = min(eval_size, max_eval_size)
    train_data, eval_data = torch.split(data, [len(data) - eval_size, eval_size])
    return train_data, eval_data