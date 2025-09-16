# In cleanil/rl/critic_discrete.py
import torch
import torch.nn as nn

import math

from cleanil.utils import get_activation
from torchrl.modules import MLP







# You need a Positional Encoding module, as Transformers don't inherently know sequence order.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, sequence_length, embedding_dim]
        """
        # We need to permute for positional encoding addition: [seq_len, batch, dim]
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2) # Permute back to [batch, seq_len, dim]


class TransformerQNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: list, activation: str,
                 sequence_length: int, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1,d_model: int= 128):
        super().__init__()
        self.sequence_length = sequence_length
        self.true_obs_dim = obs_dim // sequence_length
        # The internal dimension of the transformer, must be divisible by nhead

        # 1. An initial linear layer to project the observation into the transformer's dimension
        self.input_proj = nn.Linear(self.true_obs_dim, d_model)
        
        # 2. Positional Encoding to inject order information
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=sequence_length)
        
        # 3. The Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. The final MLP head to get Q-values
        self.q_head = MLP(
            in_features=d_model, # Input is the Transformer's output for the last token
            out_features=num_actions,
            num_cells=hidden_dims,
            activation_class=get_activation(activation)
        )

    def forward(self, obs, act=None):
        batch_size = obs.shape[0]
        obs_seq = obs.view(batch_size, self.sequence_length, self.true_obs_dim)
        
        # Project, add positional encoding, and pass through transformer
        x = self.input_proj(obs_seq)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x) # Shape: [batch, seq_len, d_model]
        
        # We use the output corresponding to the LAST observation in the history
        last_token_out = transformer_out[:, -1, :] # Shape: [batch, d_model]
        
        q_all = self.q_head(last_token_out)
        
        # ... (same logic as before for selecting action)
        if act is None:
            return q_all
        else:
            act_idx = act.squeeze(-1).long()
            q_selected = q_all.gather(1, act_idx.unsqueeze(-1))
            return q_selected

# Assuming TransformerQNetwork and other modules are defined as in the previous answer

class HistoryDoubleQNetwork_transformer(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: list, activation: str,
                 sequence_length: int, nhead: int, num_layers: int, dropout: float = 0.1,d_model: int= 128):
        """
        This wrapper now creates two independent TransformerQNetwork instances.
        """
        super().__init__()
        
        # Instantiate the first Q-network (q1)
        self.q1 = TransformerQNetwork(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            activation=activation,
            sequence_length=sequence_length,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            d_model=d_model
        )
        
        # Instantiate the second, separate Q-network (q2)
        self.q2 = TransformerQNetwork(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            activation=activation,
            sequence_length=sequence_length,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            d_model=d_model
        )
        
    def forward(self, obs, act=None):
        # The forward pass logic is unchanged.
        # It just calls the forward methods of the two underlying transformer networks.
        q1_vals = self.q1(obs, act)
        q2_vals = self.q2(obs, act)
        return q1_vals, q2_vals



# The rest of your training loop (target network creation, loss calculation)
# remains exactly the same.

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