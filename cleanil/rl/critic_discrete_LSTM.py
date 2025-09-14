# In cleanil/rl/critic_discrete.py
import torch
import torch.nn as nn
from cleanil.utils import get_activation
from torchrl.modules import MLP




class HistoryBasedQNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: list, activation: str,
                 sequence_length: int, rnn_hidden_size: int = 128, rnn_num_layers: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.true_obs_dim = obs_dim // sequence_length # Dim of a single frame
        
        self.lstm = nn.LSTM(
            input_size=self.true_obs_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True
        )
        
        # The MLP head takes the FINAL output of the LSTM
        self.q_head = MLP(
            in_features=rnn_hidden_size,
            out_features=num_actions,
            num_cells=hidden_dims,
            activation_class=get_activation(activation)
        )
    
    def forward(self, obs, act=None):
        # obs shape: [batch_size, sequence_length * true_obs_dim]
        batch_size = obs.shape[0]
        
        # 1. Reshape the flat input into a sequence for the LSTM
        # Shape becomes: [batch_size, sequence_length, true_obs_dim]
        obs_seq = obs.view(batch_size, self.sequence_length, self.true_obs_dim)
        
        # 2. Pass through LSTM. We only care about the final output.
        # lstm_out shape: [batch_size, sequence_length, rnn_hidden_size]
        lstm_out, _ = self.lstm(obs_seq) # We ignore the hidden state output
        
        # 3. Select the output from the VERY LAST time step of the sequence
        last_timestep_out = lstm_out[:, -1, :] # Shape: [batch_size, rnn_hidden_size]
        
        # 4. Get Q-values from the MLP head
        q_all = self.q_head(last_timestep_out) # Shape: [batch_size, num_actions]
        
        if act is None:
            return q_all
        else:
            act_idx = act.squeeze(-1).long()
            q_selected = q_all.gather(1, act_idx.unsqueeze(-1))
            return q_selected

# And the DoubleQNetwork wrapper remains almost the same, just passing the new args
class HistoryDoubleQNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dims, activation, sequence_length, rnn_hidden_size=128):
        super().__init__()
        self.q1 = HistoryBasedQNetwork(obs_dim, num_actions, hidden_dims, activation, sequence_length, rnn_hidden_size)
        self.q2 = HistoryBasedQNetwork(obs_dim, num_actions, hidden_dims, activation, sequence_length, rnn_hidden_size)
        
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