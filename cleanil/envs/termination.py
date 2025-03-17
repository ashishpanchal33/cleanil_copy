import torch
from cleanil.data import denormalize


class Termination:
    def __init__(self, obs_mean=0., obs_std=1.):
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def termination_fn(self, next_obs):
        raise NotImplementedError
    

class HalfCheetahTermination(Termination):
    def __init__(self, obs_mean=0., obs_std=1.):
        super().__init__(obs_mean, obs_std)

    def termination_fn(self, _obs: torch.Tensor):
        obs = denormalize(_obs, self.obs_mean, self.obs_std**2)

        """TEMP enhance stability"""
        not_done = torch.all(obs.abs() < 100, dim=-1) \
            * torch.all(_obs.abs() < 30, dim=-1)
        done = ~not_done.unsqueeze(-1)
        return done
    

class HopperTermination(Termination):
    def __init__(self, obs_mean=0., obs_std=1.):
        super().__init__(obs_mean, obs_std)
    
    def termination_fn(self, _obs: torch.Tensor):
        obs = denormalize(_obs, self.obs_mean, self.obs_std**2)
        
        height = obs[..., 0]
        angle = obs[..., 1]
        not_done = torch.all(obs.abs() < 100, dim=-1) \
            * torch.all(_obs.abs() < 30, dim=-1) \
            * (height > .7) \
            * (torch.abs(angle) < .2)

        done = ~not_done.unsqueeze(-1)
        return done
    

class WalkerTermination(Termination):
    def __init__(self, obs_mean=0., obs_std=1.):
        super().__init__(obs_mean, obs_std)

    def termination_fn(self, _obs: torch.Tensor):
        obs = denormalize(_obs, self.obs_mean, self.obs_std)

        height = obs[..., 0]
        angle = obs[..., 1]
        not_done =  torch.all(obs.abs() < 100, dim=-1) \
            * torch.all(_obs.abs() < 30, dim=-1) \
            * (height > 0.8) \
            * (height < 2.0) \
            * (angle > -1.0) \
            * (angle < 1.0)
        done = ~not_done.unsqueeze(-1)
        return done


def get_termination_fn(env_name, obs_mean=0., obs_std=1.):
    if "Hopper" in env_name:
        termination_fn = HopperTermination(obs_mean, obs_std).termination_fn
    elif "HalfCheetah" in env_name:
        termination_fn = HalfCheetahTermination(obs_mean, obs_std).termination_fn
    elif "Walker" in env_name:
        termination_fn = WalkerTermination(obs_mean, obs_std).termination_fn
    else:
        raise NotImplementedError
    return termination_fn