import torch
import torch.nn as nn
from cleanil.utils import get_activation
from torchrl.modules import MLP


class DoubleQNetwork(nn.Module):
    """Double Q network for continuous actions"""
    def __init__(self, 
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int], 
        activation: str,
    ):
        super().__init__()
        kwargs = {
            "in_features": observation_dim + action_dim,
            "out_features": 1,
            "num_cells": hidden_dims,
            "activation_class": get_activation(activation),
        }
            
        self.q1 = MLP(**kwargs)
        self.q2 = MLP(**kwargs)

    def forward(self, o, a):
        """Compute q1 and q2 values
        
        Args:
            o (torch.tensor): observation. size=[batch_size, obs_dim]
            a (torch.tensor): action. size=[batch_size, act_dim]

        Returns:
            q1 (torch.tensor): q1 value. size=[batch_size, 1]
            q2 (torch.tensor): q2 value. size=[batch_size, 1]
        """
        oa = torch.cat([o, a], dim=-1)
        q1 = self.q1(oa)
        q2 = self.q2(oa)
        return q1, q2


def compute_q_target(r, v_next, done, gamma, closed_form_terminal=False):
    q_target = r + (1 - done) * gamma * v_next
    if closed_form_terminal: # special handle terminal state
        v_done = gamma / (1 - gamma) * r
        q_target += done * v_done
    return q_target

def update_critic_target(critic: nn.Module, target: nn.Module, polyak: float):
    for p, p_target in zip(
        critic.parameters(), target.parameters()
    ):
        p_target.data.mul_(polyak)
        p_target.data.add_((1 - polyak) * p.data)