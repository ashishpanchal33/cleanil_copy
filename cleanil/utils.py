import argparse
import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union
from torchrl.record.loggers import get_logger as get_logger_torchrl
from torchrl.record.loggers import Logger
from tensordict import TensorDict

@dataclass
class LoggerConfig:
    backend: str = "wandb"
    log_dir: str = "."
    exp_name: str = ""
    project_name: str = ""
    group_name: str = ""
    wandb_mode: str = "online"


def load_yaml(from_command_line: bool = True, config_path: str = "") -> dict:
    if from_command_line:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", type=str, default="")
        args = vars(parser.parse_args())
        config_path = args["config_path"]

    assert config_path.endswith(".yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_dir(path: str):
    if path != "" and not os.path.exists(path):
        os.makedirs(path)

def get_device(device: str):
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device

def get_logger(config: dict, logger_config: LoggerConfig) -> Union[Logger | None]:
    logger = None
    if logger_config.backend:
        logger = get_logger_torchrl(
            logger_type=logger_config.backend,
            logger_name=logger_config.log_dir,
            experiment_name=logger_config.exp_name,
            wandb_kwargs={
                "mode": logger_config.wandb_mode,
                "config": config,
                "project": logger_config.project_name,
                "group": logger_config.group_name,
            },
        )
    return logger

def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "silu":
        return nn.SiLU
    else:
        raise NotImplementedError
    
def print_grads(model: torch.nn.Module):
    for n, p in model.named_parameters():
        if p.grad is None:
            print(n, None)
        else:
            print(n, p.grad.data.norm())

def remove_keys_from_tensordict(td: TensorDict, keys_to_remove: list[str]) -> TensorDict:
    """
    Recursively removes keys from a nested TensorDict.
    
    Args:
        td (TensorDict): The input TensorDict.
        keys_to_remove (set or list): The keys to remove.
        
    Returns:
        TensorDict: A new TensorDict with the specified keys removed.
    """
    # Create a new dictionary to hold the filtered fields
    filtered_fields = {}
    for key, value in td.items():
        if key in keys_to_remove:
            # Skip keys that need to be removed
            continue
        elif isinstance(value, TensorDict):
            # Recurse if the value is a nested TensorDict
            filtered_fields[key] = remove_keys_from_tensordict(value, keys_to_remove)
        else:
            # Keep the key-value pair
            filtered_fields[key] = value

    # Return a new TensorDict with the filtered fields
    return TensorDict(filtered_fields, batch_size=td.batch_size)

def concat_tensordict_on_shared_keys(td1: TensorDict, td2: TensorDict) -> TensorDict:
    # Find shared keys
    shared_keys = set(td1.keys()).intersection(set(td2.keys()))

    # Prepare concatenated TensorDict
    concatenated = {}
    for key in shared_keys:
        val1, val2 = td1[key], td2[key]
        
        # If values are TensorDicts, recurse
        if isinstance(val1, TensorDict) and isinstance(val2, TensorDict):
            concatenated[key] = concat_tensordict_on_shared_keys(val1, val2)
        # Otherwise, concatenate tensors along the batch dimension
        else:
            concatenated[key] = torch.cat([val1, val2], dim=0)

    # Determine the new batch size (assuming concatenation along batch dimension)
    new_batch_size = [td1.batch_size[0] + td2.batch_size[0]]
    return TensorDict(concatenated, batch_size=new_batch_size)

def freeze_model_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def compute_parameter_l2(model: nn.Module) -> torch.Tensor:
    loss = 0.
    for p in model.parameters():
        if p.requires_grad:
            loss += torch.sum(p ** 2) / 2.
    return loss

def compute_linear_scale(min_val: float, max_val: float, min_step: float, max_step: float, t: float) -> float:
    """Linearly increate value based on step"""
    ratio = (t - min_step) / (max_step - min_step)
    ratio = max(0, min(ratio, 1))
    val = min_val + ratio * (max_val - min_val)
    return val

def tanh_scaling(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """Map value x to interval [min_val, max_val] using tanh"""
    x = torch.tanh(x)
    x = min_val + 0.5 * (max_val - min_val) * (x + 1)  # From SpinUp / Denis Yarats
    return x

def soft_clamp(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """Soft clamp value x to interval near (min_val, max_val) using softplus"""
    x = max_val - F.softplus(max_val - x)
    x = min_val + F.softplus(x - min_val)
    return x

def clip_by_quantile(x: torch.Tensor, quantile: float = 0.95, max_std: float = 6.):
    x_clip_quantile = x.clip(torch.quantile(x, 1 - quantile), torch.quantile(x, quantile))
    clip_min = x_clip_quantile.mean(0) - x_clip_quantile.std(0) * max_std
    clip_max = x_clip_quantile.mean(0) + x_clip_quantile.std(0) * max_std
    x_clip = x.clip(clip_min, clip_max)
    return x_clip