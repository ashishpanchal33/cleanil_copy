import unittest
import torch
from cleanil.rl.actor import TanhNormalActor
from cleanil.utils import get_activation
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

class TestActor(unittest.TestCase):
    def test_actor(self):
        observation_dim = 10
        action_dim = 6
        hidden_dims = [64, 64]
        activation = "silu"

        actor_kwargs = {
            "in_features": observation_dim,
            "out_features": 2 * action_dim,
            "num_cells": hidden_dims,
            "activation_class": get_activation(activation),
        }
        dist_kwargs = {
            "low": -torch.ones(action_dim),
            "high": torch.ones(action_dim),
        }

        actor = TanhNormalActor(
            actor_kwargs, dist_kwargs,
        )

        # test sampling mode
        batch_size = 125
        obs = TensorDict({"observation": torch.randn(batch_size, observation_dim)})
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            obs = actor.forward(obs)
            loc, _ = actor.get_stats(obs)
            
        self.assertTrue(torch.allclose(obs["action"], loc.tanh(), atol=1e-6))


if __name__ == "__main__":
    unittest.main()