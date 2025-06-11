from typing import Callable
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from cleanil.utils import get_activation
from torchrl.modules import MLP
from typing import Tuple


class RewardModel(nn.Module):
    def __init__(
        self, 
        observation_dim: int, 
        action_dim: int, 
        hidden_dims: list[int], 
        activation: str,
        state_only: bool = False,
        use_done: bool = True,
        max_reward: float = 10.,
    ):
        super().__init__()
        self.state_only = state_only
        self.use_done = use_done
        self.max_reward = max_reward
        
        self.mlp = MLP(**{
            "in_features": observation_dim + action_dim * (not state_only) + 1 * use_done,
            "out_features": 1,
            "num_cells": hidden_dims,
            "activation_class": get_activation(activation),
        })

    def compute_inputs(self, obs: torch.Tensor, act: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        rwd_inputs = obs
        if not self.state_only:
            rwd_inputs = torch.cat([rwd_inputs, act], dim=-1)
        
        if self.use_done:
            rwd_inputs = torch.cat([rwd_inputs * (1 - done), done], dim=-1)
        return rwd_inputs
    
    def forward(self, obs: torch.Tensor, act: torch.Tensor, done: torch.Tensor, clip: bool = True) -> torch.Tensor:
        rwd_inputs = self.compute_inputs(obs, act, done)
        rwd = self.mlp.forward(rwd_inputs)
        if clip:
            rwd = rwd.clip(-self.max_reward, self.max_reward)
        return rwd
    
    def forward_on_inputs(self, rwd_inputs: torch.Tensor) -> torch.Tensor:
        """Save input processing"""
        return self.mlp.forward(rwd_inputs).clip(-self.max_reward, self.max_reward)
    

@torch._dynamo.disable
def compute_grad_penalty(
    real_inputs: torch.Tensor, 
    fake_inputs: torch.Tensor, 
    forward_func: Callable,
    grad_target: float,
    penalty_type: str = "margin",
    norm_ord: int | str = float("inf"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = torch.rand(len(real_inputs), 1, device=real_inputs.device)
    interpolated = alpha * real_inputs + (1 - alpha) * fake_inputs
    interpolated = Variable(interpolated, requires_grad=True)

    rwd = forward_func(interpolated)
    grad = torch_grad(
        outputs=rwd, inputs=interpolated, 
        grad_outputs=torch.ones_like(rwd),
        create_graph=True, retain_graph=True
    )[0]

    if penalty_type == "margin":
        grad_norm = torch.linalg.norm(grad, ord=norm_ord, dim=-1)
        grad_pen = torch.max(
            grad_norm - grad_target, 
            torch.zeros_like(grad_norm, device=grad_norm.device),
        ).pow(2).mean()
    else:
        grad_norm = torch.linalg.norm(grad, ord=norm_ord, dim=-1)
        grad_pen = torch.pow(grad_norm - grad_target, 2).mean()
    return grad_pen, grad_norm