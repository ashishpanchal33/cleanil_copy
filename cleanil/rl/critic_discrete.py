# In cleanil/rl/critic_discrete.py
import torch
import torch.nn as nn
from cleanil.utils import get_activation
from torchrl.modules import MLP

class DiscreteQNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: list, activation: str, dropout:float):
        super().__init__()
        self.num_actions = num_actions
        self.q_net = MLP(
            in_features=obs_dim,
            out_features=num_actions,
            num_cells=hidden_dims,
            activation_class=get_activation(activation), dropout = dropout
        )
    
    def forward(self, obs, act=None):
        """
        If act is None, return Q-values for all actions: [batch, num_actions]
        If act is provided, return Q-values for specific actions: [batch, 1]
        """
        q_all = self.q_net(obs)  # [batch, num_actions]
        
        if act is None:
            return q_all
        else:
            # act should be [batch, 1] with long dtype
            act_idx = act.squeeze(-1).long()  # [batch]
            q_selected = q_all.gather(1, act_idx.unsqueeze(-1))  # [batch, 1]
            return q_selected

class DoubleQNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: list, activation: str, dropout:float):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        
        self.q1 = DiscreteQNetwork(obs_dim, num_actions, hidden_dims, activation, dropout = dropout)
        self.q2 = DiscreteQNetwork(obs_dim, num_actions, hidden_dims, activation, dropout = dropout)
    
    def forward(self, obs, act=None):
        q1_vals = self.q1(obs, act)
        q2_vals = self.q2(obs, act)
        return q1_vals, q2_vals


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