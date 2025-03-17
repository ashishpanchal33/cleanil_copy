import unittest
import torch
from cleanil.utils import get_activation
from tensordict import TensorDict

class TestDynamics(unittest.TestCase):
    def test_ensemble_mlp(self):
        from cleanil.dynamics.ensemble_dynamics import EnsembleMLP

        input_dim = 17
        output_dim = 5
        ensemble_dim = 7
        hidden_dims = [256, 256]
        activation = get_activation("silu")
        dynamics = EnsembleMLP(
            input_dim,
            output_dim,
            ensemble_dim,
            hidden_dims,
            activation,
        )

        # synthetic data
        batch_size = 32
        x = torch.randn(batch_size, input_dim)
        x_separate = x.unsqueeze(-2).repeat_interleave(ensemble_dim, dim=-2)

        # test forward
        pred = dynamics.forward(x)
        pred_separate = dynamics.forward_separete(x_separate)

        self.assertTrue(list(pred.shape) == [batch_size, ensemble_dim, output_dim])
        self.assertTrue(list(pred_separate.shape) == [batch_size, ensemble_dim, output_dim])
        self.assertTrue(torch.allclose(pred, pred_separate))

    def test_ensemble_dynamics(self):
        from cleanil.dynamics.ensemble_dynamics import EnsembleDynamics, EnsembleConfig

        config = EnsembleConfig()
        obs_dim = 17
        act_dim = 6
        out_dim = obs_dim + 1
        ensemble_dim = config.ensemble_dim
        min_std = config.min_std
        max_std = config.max_std

        dynamics = EnsembleDynamics(
            obs_dim,
            act_dim,
            out_dim,
            config,
        )

        # synthetic data
        batch_size = 1000
        obs = torch.randn(batch_size, obs_dim)
        act = torch.randn(batch_size, act_dim)
        rwd = torch.randn(batch_size, 1)
        next_obs = torch.randn(batch_size, obs_dim)

        obs_sep = torch.randn(batch_size, ensemble_dim, obs_dim)
        act_sep = torch.randn(batch_size, ensemble_dim, act_dim)
        
        inputs = torch.cat([obs, act], dim=-1)
        targets = torch.cat([next_obs - obs, rwd], dim=-1)
        inputs_sep = torch.cat([obs_sep, act_sep], dim=-1)

        # test internal variable shapes
        self.assertTrue(list(dynamics.topk_dist.shape) == [ensemble_dim])
        self.assertTrue(list(dynamics.min_lv.shape) == [out_dim])
        self.assertTrue(list(dynamics.max_lv.shape) == [out_dim])
        self.assertTrue(torch.isclose(dynamics.min_lv.exp() ** 0.5, min_std * torch.ones(1), atol=1e-5).all())
        self.assertTrue(torch.isclose(dynamics.max_lv.exp() ** 0.5, max_std * torch.ones(1), atol=1e-5).all())
        
        # test method output shapes on regular batch
        pred, _ = dynamics.compute_stats(inputs)
        dist = dynamics.get_dist(obs, act)
        logp = dynamics.compute_log_prob(obs, act, targets)
        mix_logp = dynamics.compute_mixture_log_prob(obs, act, targets)
        sample = dynamics.sample_dist(obs, act)
        
        self.assertTrue(list(pred.shape) == [batch_size, ensemble_dim, out_dim])
        self.assertTrue(list(dist.mean.shape) == [batch_size, ensemble_dim, out_dim])
        self.assertTrue(list(dist.variance.shape) == [batch_size, ensemble_dim, out_dim])
        self.assertTrue(list(logp.shape) == [batch_size, ensemble_dim, 1])
        self.assertTrue(list(mix_logp.shape) == [batch_size, 1])
        self.assertTrue(list(sample.shape) == [batch_size, out_dim])

        # test method output shapes on separate batch
        pred_sep, _ = dynamics.compute_stats_separate(inputs_sep)
        self.assertTrue(list(pred_sep.shape) == [batch_size, ensemble_dim, out_dim])

    def test_ensemble_training(self):
        from cleanil.dynamics.ensemble_dynamics import (
            EnsembleDynamics, 
            EnsembleConfig,
            train_ensemble,
            remove_reward_head,
        )

        config = EnsembleConfig()
        obs_dim = 17
        act_dim = 6
        out_dim = obs_dim + 1
        batch_size = 64
        lr = 1e-3
        eval_ratio = 0.2
        epochs = 2

        dynamics = EnsembleDynamics(
            obs_dim,
            act_dim,
            out_dim,
            config,
        )
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=lr)

        # synthetic data
        data_size = 10000
        obs = torch.randn(data_size, obs_dim) + 10.
        act = torch.randn(data_size, act_dim) + 3.
        rwd = torch.randn(data_size, 1) - 5.
        next_obs = torch.randn(data_size, obs_dim)
        data = TensorDict(
            {
                "observation": obs,
                "action": act,
                "next": {
                    "observation": next_obs,
                    "reward": rwd,
                }
            },
            batch_size=(data_size,)
        )
        
        pred_rwd = True
        train_ensemble(
            data,
            pred_rwd,
            dynamics,
            optimizer,
            eval_ratio,
            batch_size,
            epochs,
        )

        remove_reward_head(dynamics.state_dict(), obs_dim)


if __name__ == "__main__":
    unittest.main()