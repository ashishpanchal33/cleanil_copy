import time
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import torch
import torch.nn as nn
from cleanil.base_trainer import BaseTrainer, BaseTrainerConfig
from cleanil.rl.actor import TanhNormalActor, compute_action_likelihood
from cleanil.utils import compute_parameter_l2

from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import Logger


@dataclass
class BCConfig(BaseTrainerConfig):
    # data args
    dataset: str = "halfcheetah-expert-v2"
    num_expert_trajs: int = 20
    train_ratio: float = 0.9

    # policy args
    hidden_dims: list = (256, 256)
    activation: str = "silu"
    l2_penalty: float = 0.0001
    batch_size: int = 256
    lr_actor: float = 3e-4
    grad_clip: float = 1000.

    # train args
    max_eps_steps: int = 1000
    epochs: int = 1000
    steps_per_epoch: int = 1000
    num_eval_eps: int = 1
    eval_steps: int = 1000


@torch.compile
def update_actor(
    batch: TensorDict,
    actor: nn.Module,
    l2_penalty: float,
    actor_optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    obs = batch["observation"]
    act = batch["action"]

    actor_optimizer.zero_grad()
    logp, act_pred = compute_action_likelihood(obs, act, actor)
    l2 = compute_parameter_l2(actor)
    actor_loss = -logp.mean() + l2_penalty * l2

    actor_loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    actor_optimizer.step()

    with torch.no_grad():
        mae = torch.abs(act_pred - act).mean()

    stats = {
        "act_logp": logp.mean().detach().cpu().numpy(),
        "act_mae": mae.detach().cpu().numpy(),
    }
    return stats


class Trainer(BaseTrainer):
    """Behavior cloning"""
    def __init__(
        self, 
        config: BCConfig,
        actor: TanhNormalActor,
        expert_buffer: Optional[ReplayBuffer],
        eval_buffer: Optional[ReplayBuffer],
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        eval_env: Optional[TransformedEnv] = None,
        logger: Optional[Logger] = None,
        device: torch.device = "cpu",
    ):
        super().__init__(
            config, None, eval_env, None, logger, device
        )
        self.config = config
        self.expert_buffer = expert_buffer
        self.eval_buffer = eval_buffer

        self.actor = actor

        self.optimizers = {
            "actor": torch.optim.Adam(
                self.actor.parameters(), lr=config.lr_actor
            ),
        }

        # store modules for checkpointing
        self._modules["actor"] = self.actor
        self._modules["obs_mean"] = nn.Parameter(obs_mean, requires_grad=False)
        self._modules["obs_std"] = nn.Parameter(obs_std, requires_grad=False)
    
    def take_policy_gradient_step(self, batch: TensorDict, global_step: int):
        l2_penalty = self.config.l2_penalty
        grad_clip = self.config.grad_clip

        stats = update_actor(
            batch,
            self.actor,
            l2_penalty,
            self.optimizers["actor"],
            grad_clip,
        )

        if self.logger is not None:
            for metric_name, metric_value in stats.items():
                self.logger.log_scalar(f"policy/{metric_name}", metric_value, global_step)
        return stats
    
    def train_policy_epoch(self, global_step: int):
        batch_size = self.config.batch_size

        batch = self.expert_buffer.sample(batch_size)
        policy_stats = self.take_policy_gradient_step(batch, global_step)
        return policy_stats
    
    def eval_action_likelihood(self, global_step: int, num_samples=1000):
        with torch.no_grad():
            eval_batch = self.eval_buffer.sample(num_samples)
            obs = eval_batch["observation"]
            act = eval_batch["action"]
            logp, act_pred = compute_action_likelihood(obs, act, self.actor)
            mae = torch.abs(act_pred - act).mean()

        stats = {
            "eval_act_logp": logp.mean().detach().cpu().numpy(), 
            "eval_act_mae": mae.detach().cpu().numpy(),
        }
        
        if self.logger is not None:
            for metric_name, metric_value in stats.items():
                self.logger.log_scalar(metric_name, metric_value, global_step)

    def train(self):
        config = self.config
        epochs = config.epochs
        steps_per_epoch = config.steps_per_epoch
        max_eps_steps = config.max_eps_steps
        num_eval_eps = config.num_eval_eps
        save_steps = config.save_steps

        total_steps = epochs * steps_per_epoch
        start_time = time.time()
        
        epoch = 0
        pbar = tqdm(range(total_steps))
        for t in pbar:
            policy_stats_epoch = self.train_policy_epoch(t)

            # end of epoch handeling
            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                # evaluate episodes
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    if num_eval_eps > 0:
                        eval_rollout = self.eval_env.rollout(
                            max_eps_steps,
                            self.actor,
                            auto_cast_to_device=True,
                            break_when_any_done=True,
                        )
                        eval_return = eval_rollout["next", "reward"].cpu().sum(-2).numpy().mean()
                        speed = (t + 1) / (time.time() - start_time)
                        if self.logger is not None:
                            self.logger.log_scalar("eval_eps_return", eval_return, t)
                            self.logger.log_scalar("train_speed", speed, t)
                
                desc_str = f"epoch: {epoch}/{epochs}, step: {t}, eval_eps_return: {eval_return:.2f}"
                pbar.set_description(desc_str)

                self.eval_action_likelihood(t)

                # checkpoint handling
                if save_steps not in [None, -1] and (epoch + 1) % save_steps == 0:
                    self.save()
        
        pbar.close()
        self.close_envs()
        self.save()