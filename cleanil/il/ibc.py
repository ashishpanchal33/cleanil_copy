import time
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from optree import tree_map
from cleanil.base_trainer import BaseTrainer, BaseTrainerConfig
from cleanil.rl.actor import TanhNormalActor, sample_actor, compute_action_likelihood
from cleanil.il.reward import RewardModel, compute_grad_penalty
from cleanil.utils import compute_parameter_l2

from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import Logger


@dataclass
class IBCConfig(BaseTrainerConfig):
    # data args
    dataset: str = "halfcheetah-expert-v2"
    num_expert_trajs: int = 20
    train_ratio: float = 0.9

    # reward args
    reward_loss_type: str = "nce"
    max_reward: float = 100.
    reward_state_only: bool = False
    reward_use_done: bool = False
    reward_grad_target: float = 1.
    reward_grad_penalty: float = 10.
    reward_l2_penalty: float = 1.e-5
    lr_reward: float = 3e-4
    update_reward_every: int = 1
    reward_train_steps: int = 1
    num_actor_samples: int = 1
    num_rand_samples: int = 3

    # policy args
    hidden_dims: list = (256, 256)
    activation: str = "silu"
    alpha: float = 0.2
    tune_alpha: float = True
    batch_size: int = 256
    policy_train_steps: int = 1
    lr_actor: float = 3e-4
    grad_clip: float = 1000.

    # train args
    buffer_size: int = 1e6
    max_eps_steps: int = 1000
    epochs: int = 1000
    steps_per_epoch: int = 1000
    num_eval_eps: int = 1
    eval_steps: int = 1000


def compute_actor_loss(
    batch: TensorDict,
    reward: RewardModel,
    actor: nn.Module,
    alpha: torch.Tensor,
):
    obs = batch["observation"]
    act, logp = sample_actor(obs, actor)
    
    rwd_inputs = reward.compute_inputs(
        obs, act, batch["terminated"].float()
    )
    q = reward.forward_on_inputs(rwd_inputs)

    actor_loss = torch.mean(alpha * logp - q)
    
    stats = {
        "actor_loss": actor_loss.detach().cpu().numpy(),
        "alpha": alpha[0].detach().cpu().numpy(),
    }
    return actor_loss, logp.data, stats

def compute_alpha_loss(
    logp: torch.Tensor,
    log_alpha: torch.Tensor,
    alpha_target: torch.Tensor,
):
    alpha_loss = -torch.mean(log_alpha.exp() * (logp + alpha_target).detach()) # cleanrl implementation
    
    stats = {"alpha_loss": alpha_loss.detach().cpu().numpy()}
    return alpha_loss, stats

@torch.compile
def update_actor(
    batch: TensorDict,
    reward: RewardModel,
    actor: nn.Module,
    alpha: torch.Tensor,
    log_alpha: torch.Tensor,
    alpha_target: torch.Tensor,
    tune_alpha: bool,
    actor_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    actor_optimizer.zero_grad()
    actor_loss, logp, actor_stats = compute_actor_loss(
            batch, reward, actor, alpha
        )
    actor_loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    actor_optimizer.step()

    alpha_loss = torch.zeros(1, device=logp.device)
    if tune_alpha:
        alpha_optimizer.zero_grad()
        alpha_loss, alpha_stats = compute_alpha_loss(logp, log_alpha, alpha_target)
        alpha_loss.backward()
        alpha_optimizer.step()

        actor_stats.update(alpha_stats)
    return actor_stats

def compute_reward_loss(
    real_batch: TensorDict,
    fake_batch: TensorDict,
    reward: RewardModel,
    grad_target: float,
    grad_penalty: float,
    l2_penalty: float,
):
    """Fake batch has dimension [num_samples, batch_size, ...]"""
    real_rwd_inputs = reward.compute_inputs(
        real_batch["observation"], real_batch["action"], real_batch["terminated"].float()
    )
    fake_rwd_inputs = reward.compute_inputs(
        fake_batch["observation"], fake_batch["action"], fake_batch["terminated"].float()
    )
    real_rwd = reward.forward_on_inputs(real_rwd_inputs)
    fake_rwd = reward.forward_on_inputs(fake_rwd_inputs)
    reward_loss = -torch.mean(real_rwd - fake_rwd.mean(0))

    num_samples = fake_batch.shape[0]
    _real_rwd_inputs = torch.cat([real_rwd_inputs for _ in range(num_samples)], dim=0)
    _fake_rwd_inputs = fake_rwd_inputs.flatten(0, 1)
    gp, g_norm = compute_grad_penalty(
        _real_rwd_inputs, 
        _fake_rwd_inputs, 
        reward.forward_on_inputs, 
        grad_target,
        penalty_type="margin",
        norm_ord=float("inf"),
    )
    l2 = compute_parameter_l2(reward)
    total_loss = reward_loss + grad_penalty * gp + l2_penalty * l2

    with torch.no_grad():
        reward_abs = (real_rwd.abs().mean() + fake_rwd.abs().mean()) / 2

    stats = {
        "reward_loss": reward_loss.detach().cpu().numpy(),
        "grad_penalty": gp.detach().cpu().numpy(),
        "grad_norm": g_norm.detach().cpu().numpy().mean(),
        "reward_abs": reward_abs.detach().cpu().numpy(),
        "reward_l2": l2.detach().cpu().numpy(),
    }
    return total_loss, stats

def sample_random_actions(batch: TensorDict, num_samples: int):
    min_act = batch["action"].min(0)[0]
    max_act = batch["action"].max(0)[0]
    rand_batch = torch.stack([batch.clone() for _ in range(num_samples)], dim=0)
    rand_act = torch.distributions.Uniform(min_act, max_act).sample(rand_batch["action"].shape[:-1])
    rand_batch["action"] = rand_act
    return rand_batch

def compute_reward_loss_nce(
    real_batch: TensorDict,
    fake_batch: TensorDict,
    rand_batch: TensorDict,
    reward: RewardModel,
    grad_target: float,
    grad_penalty: float,
    l2_penalty: float,
):
    """Fake and rand batch have dimension [num_samples, batch_size, ...]"""
    real_rwd_inputs = reward.compute_inputs(
        real_batch["observation"], real_batch["action"], real_batch["terminated"].float()
    )
    fake_rwd_inputs = reward.compute_inputs(
        fake_batch["observation"], fake_batch["action"], fake_batch["terminated"].float()
    )
    rand_rwd_inputs = reward.compute_inputs(
        rand_batch["observation"], rand_batch["action"], rand_batch["terminated"].float()
    )
    real_rwd = reward.forward_on_inputs(real_rwd_inputs)
    fake_rwd = reward.forward_on_inputs(fake_rwd_inputs)
    rand_rwd = reward.forward_on_inputs(rand_rwd_inputs)

    # compute info nce loss
    rwds = torch.cat([real_rwd.unsqueeze(0), fake_rwd, rand_rwd], dim=0)
    log_z = torch.logsumexp(rwds, dim=0)
    reward_loss = -torch.mean(real_rwd - log_z)

    # compute gradient penalty
    num_samples = fake_batch.shape[0] + rand_batch.shape[0]
    _real_rwd_inputs = torch.cat([real_rwd_inputs for _ in range(num_samples)], dim=0)
    _fake_rwd_inputs = torch.cat([fake_rwd_inputs.flatten(0, 1), rand_rwd_inputs.flatten(0, 1)], dim=0)
    gp, g_norm = compute_grad_penalty(
        _real_rwd_inputs, 
        _fake_rwd_inputs, 
        reward.forward_on_inputs, 
        grad_target,
        penalty_type="margin",
        norm_ord=float("inf"),
    )
    l2 = compute_parameter_l2(reward)
    total_loss = reward_loss + grad_penalty * gp + l2_penalty * l2

    with torch.no_grad():
        reward_abs = (real_rwd.abs().mean() + fake_rwd.abs().mean()) / 2
        acc = torch.mean((rwds.argmax(0) == 0).float())
        acc_topk = torch.mean((rwds.argmax(0) <= fake_batch.shape[0]).float())

    stats = {
        "reward_loss": reward_loss.detach().cpu().numpy(),
        "grad_penalty": gp.detach().cpu().numpy(),
        "grad_norm": g_norm.detach().cpu().numpy().mean(),
        "reward_abs": reward_abs.detach().cpu().numpy(),
        "reward_l2": l2.detach().cpu().numpy(),
        "reward_accuracy": acc.detach().cpu().numpy(),
        "reward_accuracy_topk": acc_topk.detach().cpu().numpy(),
        "real_reward_mean": real_rwd.detach().mean().cpu().numpy(),
        "fake_reward_mean": fake_rwd.detach().mean().cpu().numpy(),
        "rand_reward_mean": rand_rwd.detach().mean().cpu().numpy(),
    }
    return total_loss, stats

@torch.compile
def update_reward(
    real_batch: TensorDict,
    fake_batch: TensorDict,
    rand_batch: TensorDict,
    reward: RewardModel,
    loss_type: str,
    grad_target: float,
    grad_penalty: float,
    l2_penalty: float,
    reward_optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    reward_optimizer.zero_grad()
    if loss_type == "nce":
        reward_loss, reward_stats = compute_reward_loss_nce(
            real_batch,
            fake_batch,
            rand_batch,
            reward,
            grad_target,
            grad_penalty,
            l2_penalty,
        )
    else:
        reward_loss, reward_stats = compute_reward_loss(
            real_batch,
            fake_batch,
            reward,
            grad_target,
            grad_penalty,
            l2_penalty,
        )
    reward_loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(reward.parameters(), grad_clip)
    reward_optimizer.step()
    return reward_stats


class Trainer(BaseTrainer):
    """Implicit behavior cloning

    Using soft actor critic instread of Langevin dynamics to choose actions.

    [1] Implicit Behavioral Cloning, Florence et al, 2021 (https://arxiv.org/abs/2109.00137)
    """
    def __init__(
        self, 
        config: IBCConfig,
        actor: TanhNormalActor,
        reward: RewardModel,
        expert_buffer: ReplayBuffer,
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

        self.reward = reward
        self.actor = actor
        self.log_alpha = nn.Parameter(
            np.log(config.alpha) * torch.ones(1, device=device), 
            requires_grad=config.tune_alpha,
        )
        self.alpha = self.log_alpha.data.exp()
        self.alpha_target = -torch.prod(torch.tensor(self.eval_env.action_spec.shape, device=device))

        self.optimizers = {
            "actor": torch.optim.Adam(
                self.actor.parameters(), lr=config.lr_actor
            ),
            "alpha": torch.optim.Adam(
                [self.log_alpha], lr=config.lr_actor
            ),
            "reward": torch.optim.Adam(
                self.reward.parameters(), lr=config.lr_reward
            ),
        }

        # store modules for checkpointing
        self._modules["actor"] = self.actor
        self._modules["log_alpha"] = self.log_alpha
        self._modules["reward"] = self.reward
        self._modules["obs_mean"] = nn.Parameter(obs_mean, requires_grad=False)
        self._modules["obs_std"] = nn.Parameter(obs_std, requires_grad=False)
    
    def take_policy_gradient_step(self, batch: TensorDict, global_step: int):
        alpha = self.alpha
        tune_alpha = self.config.tune_alpha
        grad_clip = self.config.grad_clip

        actor_stats = update_actor(
            batch,
            self.reward,
            self.actor,
            alpha,
            self.log_alpha,
            self.alpha_target,
            tune_alpha,
            self.optimizers["actor"],
            self.optimizers["alpha"],
            grad_clip,
        )

        self.alpha = self.log_alpha.detach().exp()

        if self.logger is not None:
            for metric_name, metric_value in actor_stats.items():
                self.logger.log_scalar(f"policy/{metric_name}", metric_value, global_step)
        return actor_stats
    
    def train_policy_epoch(self, global_step: int):
        train_steps = self.config.policy_train_steps
        batch_size = self.config.batch_size

        policy_stats_epoch = []
        for _ in range(train_steps):
            batch = self.expert_buffer.sample(batch_size)
            policy_stats = self.take_policy_gradient_step(batch, global_step)
            policy_stats_epoch.append(policy_stats)

        policy_stats_epoch = tree_map(lambda *x: np.stack(x).mean(0), *policy_stats_epoch)
        return policy_stats_epoch
    
    def take_reward_gradient_step(self, real_batch: TensorDict, fake_batch: TensorDict, rand_batch: TensorDict, global_step: int):
        reward_loss_type = self.config.reward_loss_type
        grad_target = self.config.reward_grad_target
        grad_penalty = self.config.reward_grad_penalty
        l2_penalty = self.config.reward_l2_penalty
        grad_clip = self.config.grad_clip

        reward_stats = update_reward(
            real_batch,
            fake_batch,
            rand_batch,
            self.reward,
            reward_loss_type,
            grad_target,
            grad_penalty,
            l2_penalty,
            self.optimizers["reward"],
            grad_clip,
        )

        if self.logger is not None:
            for metric_name, metric_value in reward_stats.items():
                self.logger.log_scalar(f"reward/{metric_name}", metric_value, global_step)
        return reward_stats
    
    def train_reward_epoch(self, global_step: int):
        train_steps = self.config.reward_train_steps
        batch_size = self.config.batch_size
        num_actor_samples = self.config.num_actor_samples
        num_rand_samples = self.config.num_rand_samples

        reward_stats_epoch = []
        for _ in range(train_steps):
            real_batch = self.expert_buffer.sample(batch_size)
            fake_batch = real_batch.clone()
            fake_batch = torch.stack([fake_batch for _ in range(num_actor_samples)], dim=0)
            with torch.no_grad():
                fake_batch = self.actor(fake_batch)
            rand_batch = sample_random_actions(real_batch, num_rand_samples)
            reward_stats = self.take_reward_gradient_step(real_batch, fake_batch, rand_batch, global_step)
            reward_stats_epoch.append(reward_stats)

        reward_stats_epoch = tree_map(lambda *x: np.stack(x).mean(0), *reward_stats_epoch)
        return reward_stats_epoch
    
    def eval_action_likelihood(self, global_step: int, num_samples=1000):
        def eval_func(batch):
            obs = batch["observation"]
            act = batch["action"]
            logp, act_pred = compute_action_likelihood(obs, act, self.actor)
            mae = torch.abs(act_pred - act).mean()
            return logp, mae
        
        with torch.no_grad():
            train_batch = self.expert_buffer.sample(num_samples)
            eval_batch = self.eval_buffer.sample(num_samples)
            train_logp, train_mae = eval_func(train_batch)
            eval_logp, eval_mae = eval_func(eval_batch)

        stats = {
            "train_act_logp": train_logp.mean().detach().cpu().numpy(), 
            "train_act_mae": train_mae.detach().cpu().numpy(),
            "eval_act_logp": eval_logp.mean().detach().cpu().numpy(), 
            "eval_act_mae": eval_mae.detach().cpu().numpy(),
        }
        
        if self.logger is not None:
            for metric_name, metric_value in stats.items():
                self.logger.log_scalar(metric_name, metric_value, global_step)

    
    def train(self):
        config = self.config
        epochs = config.epochs
        steps_per_epoch = config.steps_per_epoch
        update_reward_every = config.update_reward_every
        max_eps_steps = config.max_eps_steps
        num_eval_eps = config.num_eval_eps
        save_steps = config.save_steps

        total_steps = epochs * steps_per_epoch
        start_time = time.time()
        
        epoch = 0
        pbar = tqdm(range(total_steps))
        for t in pbar:
            policy_stats_epoch = self.train_policy_epoch(t)

            if (t + 1) % update_reward_every == 0:
                reward_stats_epoch = self.train_reward_epoch(t)

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
                
                    self.eval_action_likelihood(t)
                
                desc_str = f"epoch: {epoch}/{epochs}, step: {t}, eval_eps_return: {eval_return:.2f}"
                pbar.set_description(desc_str)

                # checkpoint handling
                if save_steps not in [None, -1] and (epoch + 1) % save_steps == 0:
                    self.save()
        
        pbar.close()
        self.close_envs()
        self.save()