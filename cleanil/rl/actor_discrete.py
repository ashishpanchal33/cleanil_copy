import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from cleanil.utils import get_activation
from torchrl.modules import MLP
from tensordict import TensorDict

class CategoricalActor(nn.Module):
    def __init__(self, mlp_config: dict):
        super().__init__()
        self.mlp = MLP(**mlp_config)

    def forward(self, obs: TensorDict):
        logits = self.mlp(obs["observation"])
        dist = Categorical(logits=logits)
        act = dist.sample()
        obs["action"] = act
        return obs
    
    def get_dist(self, obs: TensorDict):
        logits = self.mlp(obs["observation"])
        return Categorical(logits=logits)




#obs_tensor = batch["observation"]  # [batch_size, obs_dim]
#obs_dict = TensorDict({"observation": obs_tensor}, batch_size=[obs_tensor.shape[0]])
#logits = self.mlp(obs_dict["observation"]) 


def make_categorical_actor(
    observation_dim: int, 
    action_dim: int,  # number of discrete actions
    hidden_dims: list[int], 
    activation: str,
):
    actor_kwargs = {
        "in_features": observation_dim,
        "out_features": action_dim,
        "num_cells": hidden_dims,
        "activation_class": get_activation(activation),
    }
    return CategoricalActor(actor_kwargs)

def sample_discrete_actor(obs: torch.Tensor, actor: CategoricalActor):
    """Sample from discrete actor and return log likelihood"""
    actor_input = TensorDict({"observation": obs} , batch_size=[obs.shape[0]])
    dist = actor.get_dist(actor_input)
    act = dist.sample()
    logp = dist.log_prob(act).unsqueeze(-1)
    return act.unsqueeze(-1), logp  # Add dim to match continuous format

def compute_discrete_action_likelihood(
    obs: torch.Tensor, 
    act: torch.Tensor, 
    actor: CategoricalActor, 
    sample: bool = True,
):
    """Compute action likelihood for discrete actions"""
    actor_input = TensorDict({"observation": obs}, batch_size=[obs.shape[0]])
    dist = actor.get_dist(actor_input)
    
    # Convert continuous actions back to discrete if needed
    act_discrete = act.squeeze(-1).long()  # Remove last dim and convert to long
    logp = dist.log_prob(act_discrete).unsqueeze(-1)
    
    act_pred = None
    if sample:
        act_pred = dist.sample().unsqueeze(-1).float()  # Match continuous format
    return logp, act_pred
