import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.distributions as torch_dist
import torch.distributions.transforms as torch_transform
from cleanil.utils import get_activation
from tensordict import TensorDict
from tensordict.nn.probabilistic import interaction_type
from torchrl.data.tensor_specs import BoundedContinuous
from torchrl.modules import MLP
# from torchrl.modules import MLP, ProbabilisticActor
# from torchrl.modules.distributions import TanhNormal
# from tensordict.nn.distributions import NormalParamExtractor
# from tensordict.nn import InteractionType, TensorDictModule

LOG_STD_MAX = np.log(3)
LOG_STD_MIN = np.log(1e-3)

def tanh_scale_mapping(ls: torch.Tensor):
    ls = torch.tanh(ls)
    ls = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (ls + 1)  # From SpinUp / Denis Yarats
    return ls.exp()


class TanhScaleMapping(nn.Module):
    """Mapping to Normal distribution scale parameter
    Used with TensorDict NormalParamExtractor
    """
    def forward(self, ls):
        return tanh_scale_mapping(ls)


class NormalLocClipper(nn.Module):
    """Clip Normal loc paramter to ensure Tanh transform does not overflow"""
    def forward(self, tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        loc, scale = tensors
        
        eps = torch.finfo(loc.dtype).resolution
        bound = torch.atanh((1 - eps) * torch.ones(1, device = loc.device))
        loc = loc.clip(-bound, bound)
        return loc, scale


class TanhTransform(torch_transform.Transform):
    """Adapted from pytorch implementation with clipping"""
    domain = torch_dist.constraints.real
    codomain = torch_dist.constraints.real
    bijective = True
    event_dim = 0
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__() 
        self.mid = (low + high) / 2
        self.limit = (high - low) / 2
        self.eps = torch.finfo(low.dtype).resolution

    def __call__(self, x: torch.Tensor):
        y = torch.tanh(x)
        y = self.limit * y + self.mid
        return y
    
    def _inverse(self, y: torch.Tensor):
        x = (y - self.mid) / self.limit
        y = torch.clip(x, -1. + self.eps, 1. - self.eps) # prevent overflow
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        ldj = (2. * (math.log(2.) - x - F.softplus(-2. * x)))
        ldj += torch.abs(self.limit).log()
        return ldj
    

class TanhNormalActor(nn.Module):
    def __init__(self, mlp_config: dict, dist_config: dict):
        super().__init__()
        self.mlp = MLP(**mlp_config)
        self.normal_clipper = NormalLocClipper()
        self.tanh_transform = TanhTransform(
            dist_config["low"], dist_config["high"]
        )

    @torch.compile
    def forward(self, obs: TensorDict):
        sampling_mode = interaction_type()
        sampling_mode = sampling_mode.name if sampling_mode is not None else "None"
        
        if sampling_mode == "DETERMINISTIC":
            loc, _ = self.get_stats(obs)
            act = torch.tanh(loc)
        else:
            dist = self.get_dist(obs)
            act = dist.rsample()
        obs["action"] = act
        return obs
    
    def get_stats(self, obs: TensorDict):
        """Get base distribution stats"""
        loc, ls = torch.chunk(self.mlp(obs["observation"]), 2, dim=-1)
        scale = tanh_scale_mapping(ls)
        loc, scale = self.normal_clipper((loc, scale))
        return loc, scale
    
    def get_dist(self, obs: TensorDict):
        loc, scale = self.get_stats(obs)
        base_dist = torch_dist.Normal(loc, scale)
        out_dist = torch_dist.TransformedDistribution(
            base_dist, [self.tanh_transform]
        )
        return out_dist


def make_tanh_normal_actor(
    observation_dim: int, 
    action_dim: int, 
    hidden_dims: list[int], 
    activation: str,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
):
    actor_kwargs = {
        "in_features": observation_dim,
        "out_features": 2 * action_dim,
        "num_cells": hidden_dims,
        "activation_class": get_activation(activation),
    }
    dist_kwargs = {
        "low": action_low,
        "high": action_high,
    }

    actor = TanhNormalActor(actor_kwargs, dist_kwargs)
    return actor

def sample_actor(obs: torch.Tensor, actor: TanhNormalActor):
    """Reparameterized sample actor and return log likelihood"""
    actor_input = TensorDict({"observation": obs})
    dist = actor.get_dist(actor_input)
    act = dist.rsample()
    logp = dist.log_prob(act).sum(-1, keepdim=True)
    return act, logp

def compute_action_likelihood(
    obs: torch.Tensor, 
    act: torch.Tensor, 
    actor: TanhNormalActor, 
    sample: bool = True,
):
    """Compute action likelihood and return actor prediction"""
    actor_input = TensorDict({"observation": obs})
    dist = actor.get_dist(actor_input)
    logp = dist.log_prob(act).sum(-1, keepdim=True)

    act_pred = None
    if sample:
        act_pred = dist.rsample()
    return logp, act_pred

# def make_tanh_normal_actor(
#     observation_spec: BoundedContinuous, 
#     action_spec: BoundedContinuous, 
#     hidden_dims: list[int], 
#     activation: str,
# ):
#     observation_dim = observation_spec.shape[-1]
#     action_dim = action_spec.shape[-1]
#     actor_kwargs = {
#         "in_features": observation_dim,
#         "out_features": 2 * action_dim,
#         "num_cells": hidden_dims,
#         "activation_class": get_activation(activation),
#     }
#     dist_kwargs = {
#         "low": action_spec.space.low,
#         "high": action_spec.space.high,
#         "tanh_loc": False,
#         "safe_tanh": True,
#     }

#     actor_net = MLP(**actor_kwargs)
#     actor_extractor = NormalParamExtractor()
#     actor_extractor.scale_mapping = TanhScaleMapping() # force new scale mapping
#     loc_clipper = NormalLocClipper()
#     actor_net = nn.Sequential(actor_net, actor_extractor, loc_clipper)
#     actor_module = TensorDictModule(
#         actor_net,
#         in_keys=["observation"],
#         out_keys=["loc", "scale"],
#     )
#     dist_class = TanhNormal
#     actor = ProbabilisticActor(
#         spec=action_spec,
#         in_keys=["loc", "scale"],
#         module=actor_module,
#         distribution_class=dist_class,
#         distribution_kwargs=dist_kwargs,
#         default_interaction_type=InteractionType.RANDOM,
#         return_log_prob=True,
#     )
#     return actor

# def sample_actor(obs: torch.Tensor, actor: ProbabilisticActor):
#     """Reparameterized sample actor and return log likelihood"""
#     actor_input = TensorDict({"observation": obs})
#     dist = actor.get_dist(actor_input)
#     act = dist.rsample()
#     logp = dist.log_prob(act).unsqueeze(-1)
#     return act, logp

# def compute_action_likelihood(obs: torch.Tensor, act: torch.Tensor, actor: ProbabilisticActor, sample: bool = True):
#     """Compute action likelihood and return actor prediction"""
#     actor_input = TensorDict({"observation": obs})
#     dist = actor.get_dist(actor_input)
#     logp = dist.log_prob(act).unsqueeze(-1)

#     act_pred = None
#     if sample:
#         act_pred = dist.rsample()
#     return logp, act_pred