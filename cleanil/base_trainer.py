import os
from typing import Optional
from collections import OrderedDict
from dataclasses import dataclass
import torch
import torch.nn as nn
from torchrl.envs import TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.record.loggers import Logger
from cleanil.utils import make_dir

@dataclass
class BaseTrainerConfig:
    num_eval_eps: int = 10
    eval_steps: int = 1000
    save_path: str = ""
    save_steps: int = 1
    

class BaseTrainer:
    def __init__(
        self,
        config: BaseTrainerConfig,
        train_env: Optional[TransformedEnv] = None,
        eval_env: Optional[TransformedEnv] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        logger: Optional[Logger] = None,
        device: torch.device = "cpu",
    ):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.device = device

        self._modules = {}

        make_dir(self.config.save_path)

    def train(self):
        raise NotImplementedError
    
    def state_dict(self):
        state_dict = {}
        for k, v in self._modules.items():
            if isinstance(v, nn.Module):
                state_dict[k] = v.state_dict()
            elif isinstance(v, nn.Parameter):
                state_dict[k] = v.data
            else:
                raise ValueError(f"Trainer module {k} of type {type(k)} is not acceptable")
        return state_dict
    
    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            if isinstance(v, OrderedDict):
                self._modules[k].load_state_dict(v)
            elif isinstance(v, torch.Tensor):
                self._modules[k].data = v
            else:
                raise ValueError(f"state_dict module {k} of type {type(k)} is not acceptable")
    
    def save(self, path: str | None = None):
        save_path = self.config.save_path
        if path is not None:
            save_path = path
        torch.save(self.state_dict(), os.path.join(save_path, "model.p"))
    
    def close_envs(self):
        if self.eval_env is not None:
            if not self.eval_env.is_closed:
                self.eval_env.close()
        
        if self.train_env is not None:
            if not self.train_env.is_closed:
                self.train_env.close()