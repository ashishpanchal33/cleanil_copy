import time
import copy
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from optree import tree_map
from cleanil.base_trainer import BaseTrainer, BaseTrainerConfig
from cleanil.rl.critic import DoubleQNetwork, compute_q_target, update_critic_target
from cleanil.rl.actor import TanhNormalActor, sample_actor, compute_action_likelihood
from cleanil.il.reward import RewardModel, compute_grad_penalty
from cleanil.utils import (
    freeze_model_parameters, 
    concat_tensordict_on_shared_keys, 
    compute_parameter_l2,
)

from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import Logger


@dataclass
class GAILConfig(BaseTrainerConfig):
    # data args
    dataset: str = "halfcheetah-expert-v2"
    num_expert_trajs: int = 10

    # reward args
    max_reward: float = 10.
    reward_state_only: bool = False
    reward_use_done: bool = True
    reward_grad_target: float = 1.
    reward_grad_penalty: float = 1.
    reward_l2_penalty: float = 1.e-5
    lr_reward: float = 3e-4
    reward_train_steps: int = 1
    reward_update_every: int = 1

    # rl args
    hidden_dims: list = (256, 256)
    activation: str = "relu"
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    tune_alpha: float = True
    batch_size: int = 256
    real_ratio: float = 0.5
    policy_train_steps: int = 1
    policy_update_every: int = 1
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    grad_clip: float = 100.

    # train args
    buffer_size: int = 1e6
    max_eps_steps: int = 1000
    epochs: int = 1000
    steps_per_epoch: int = 1000
    update_after: int = 2000
    num_eval_eps: int = 1
    eval_steps: int = 1000


def compute_critic_loss(
    batch: TensorDict, 
    critic: nn.Module, 
    critic_target: nn.Module, 
    actor: nn.Module, 
    gamma: float, 
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    obs = batch["observation"]
    act = batch["action"]
    rwd = batch["reward"]
    done = batch["next"]["terminated"].float()
    next_obs = batch["next"]["observation"]

    with torch.no_grad():
        # compute value target
        next_act, logp = sample_actor(next_obs, actor)
        q1_next, q2_next = critic_target(next_obs, next_act)
        q_next = torch.min(q1_next, q2_next)
        v_next = q_next - alpha * logp
        q_target = compute_q_target(
            rwd, v_next, done, gamma, closed_form_terminal=False
        )

    q1, q2 = critic(obs, act)
    q1_loss = torch.pow(q1 - q_target, 2).mean()
    q2_loss = torch.pow(q2 - q_target, 2).mean()
    q_loss = (q1_loss + q2_loss) / 2
    
    stats = {"critic_loss": q_loss.detach().cpu().numpy()}
    return q_loss, stats

@torch.compile
def update_critic(
    batch: TensorDict, 
    critic: nn.Module, 
    critic_target: nn.Module, 
    actor: nn.Module, 
    gamma: float, 
    alpha: float | torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    optimizer.zero_grad()
    loss, stats = compute_critic_loss(
        batch, critic, critic_target, actor, gamma, alpha
    )
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optimizer.step()
    return stats

def compute_actor_loss(
    batch: TensorDict,
    critic: nn.Module,
    actor: nn.Module,
    alpha: torch.Tensor,
):
    obs = batch["observation"]
    act, logp = sample_actor(obs, actor)
    
    q1, q2 = critic(obs, act)
    q = torch.min(q1, q2)

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
    critic: nn.Module,
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
        batch, critic, actor, alpha
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
    real_rwd_inputs = reward.compute_inputs(
        real_batch["observation"], real_batch["action"], real_batch["next"]["terminated"].float()
    )
    fake_rwd_inputs = reward.compute_inputs(
        fake_batch["observation"], fake_batch["action"], fake_batch["next"]["terminated"].float()
    )
    real_rwd = reward.forward_on_inputs(real_rwd_inputs)
    fake_rwd = reward.forward_on_inputs(fake_rwd_inputs)
    reward_loss = -(real_rwd.mean() - fake_rwd.mean())
    gp, g_norm = compute_grad_penalty(
        real_rwd_inputs, 
        fake_rwd_inputs, 
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
        "l2": l2.detach().cpu().numpy(),
    }
    return total_loss, stats

@torch.compile
def update_reward(
    real_batch: TensorDict,
    fake_batch: TensorDict,
    reward: RewardModel,
    grad_target: float,
    grad_penalty: float,
    l2_penalty: float,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    optimizer.zero_grad()
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
    optimizer.step()
    return reward_stats


class Trainer(BaseTrainer):
    """Adversarial imitation learning with Wasserstein discriminator
    
    This algorithm implements the standard adversarial imitation learning loop with a Wasserstein discriminator.
    This is a combination of best practices in Generative Adversarial Imitation Learning [1], Discriminator Actor Critic [2], and Maximum Likelihood Inverse Reinforcement Learning [3].
    These include reward gradient penalty, terminal state handling, and optionally slow (two-timescale) reward updating.
    
    [1] Generative adversarial imitation learning, Ho et al, 2016 (https://arxiv.org/abs/1606.03476)
    [2] Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning, Kostrikov et al, 2018 (https://arxiv.org/abs/1809.02925)
    [3] Maximum-likelihood inverse reinforcement learning with finite-time guarantees, Zeng et al, 2022 (https://openreview.net/forum?id=zbt3VmTsRIj)
    """
    def __init__(
        self, 
        config: GAILConfig,
        actor: TanhNormalActor,
        critic: DoubleQNetwork,
        reward: RewardModel,
        expert_buffer: ReplayBuffer,
        train_env: TransformedEnv,
        eval_env: Optional[TransformedEnv] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        logger: Optional[Logger] = None,
        device: torch.device = "cpu",
    ):
        super().__init__(
            config, train_env, eval_env, replay_buffer, logger, device
        )
        self.config = config
        self.expert_buffer = expert_buffer

        self.reward = reward
        self.actor = actor
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = nn.Parameter(
            np.log(config.alpha) * torch.ones(1, device=device), 
            requires_grad=config.tune_alpha,
        )
        self.alpha = self.log_alpha.data.exp()
        self.alpha_target = -torch.prod(torch.tensor(self.train_env.action_spec.shape, device=device))

        freeze_model_parameters(self.critic_target)

        self.optimizers = {
            "actor": torch.optim.Adam(
                self.actor.parameters(), lr=config.lr_actor
            ),
            "critic": torch.optim.Adam(
                self.critic.parameters(), lr=config.lr_critic
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
        self._modules["critic"] = self.critic
        self._modules["critic_target"] = self.critic_target
        self._modules["log_alpha"] = self.log_alpha
        self._modules["reward"] = self.reward
    
    def take_policy_gradient_step(self, batch: TensorDict, global_step: int):
        gamma = self.config.gamma
        alpha = self.alpha
        tune_alpha = self.config.tune_alpha
        grad_clip = self.config.grad_clip

        # replace data with learned reward
        with torch.no_grad():
            batch["reward"] = self.reward.forward(
                batch["observation"], 
                batch["action"], 
                batch["next"]["terminated"].float(),
            )

        critic_stats = update_critic(
            batch,
            self.critic,
            self.critic_target,
            self.actor,
            gamma,
            alpha,
            self.optimizers["critic"],
            grad_clip,
        )

        actor_stats = update_actor(
            batch,
            self.critic,
            self.actor,
            alpha,
            self.log_alpha,
            self.alpha_target,
            tune_alpha,
            self.optimizers["actor"],
            self.optimizers["alpha"],
            grad_clip,
        )

        update_critic_target(self.critic, self.critic_target, self.config.polyak)
        self.alpha = self.log_alpha.detach().exp()
        
        metrics = {**critic_stats, **actor_stats}

        if self.logger is not None:
            for metric_name, metric_value in metrics.items():
                self.logger.log_scalar(f"policy/{metric_name}", metric_value, global_step)
        return metrics
    
    def train_policy_epoch(self, global_step: int):
        train_steps = self.config.policy_train_steps
        batch_size = self.config.batch_size
        real_ratio = self.config.real_ratio

        fake_batch_size = batch_size
        if real_ratio > 0:
            real_batch_size = int(batch_size * real_ratio)
            fake_batch_size -= real_batch_size

        policy_stats_epoch = []
        for _ in range(train_steps):
            batch = self.replay_buffer.sample(fake_batch_size)
            if real_batch_size > 0:
                real_batch = self.expert_buffer.sample(real_batch_size)
                batch = concat_tensordict_on_shared_keys(real_batch, batch)
                
            policy_stats = self.take_policy_gradient_step(batch, global_step)
            policy_stats_epoch.append(policy_stats)

        policy_stats_epoch = tree_map(lambda *x: np.stack(x).mean(0), *policy_stats_epoch)
        return policy_stats_epoch
    
    def take_reward_gradient_step(self, real_batch: TensorDict, fake_batch: TensorDict, global_step: int):
        grad_target = self.config.reward_grad_target
        grad_penalty = self.config.reward_grad_penalty
        l2_penalty = self.config.reward_l2_penalty
        grad_clip = self.config.grad_clip

        reward_stats = update_reward(
            real_batch,
            fake_batch,
            self.reward,
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

        reward_stats_epoch = []
        for _ in range(train_steps):
            real_batch = self.expert_buffer.sample(batch_size)
            fake_batch = self.replay_buffer.sample(batch_size)
            reward_stats = self.take_reward_gradient_step(real_batch, fake_batch, global_step)
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
            train_logp, train_mae = eval_func(train_batch)

        stats = {
            "train_act_logp": train_logp.mean().detach().cpu().numpy(), 
            "train_act_mae": train_mae.detach().cpu().numpy(),
        }
        
        if self.logger is not None:
            for metric_name, metric_value in stats.items():
                self.logger.log_scalar(metric_name, metric_value, global_step)
    
    def train(self):
        config = self.config
        epochs = config.epochs
        steps_per_epoch = config.steps_per_epoch
        update_after = config.update_after
        policy_update_every = config.policy_update_every
        reward_update_every = config.reward_update_every
        max_eps_steps = config.max_eps_steps
        num_eval_eps = config.num_eval_eps
        save_steps = config.save_steps

        action_spec = self.train_env.action_spec

        total_steps = epochs * steps_per_epoch + update_after
        start_time = time.time()
        
        epoch = 0
        pbar = tqdm(range(total_steps))
        eps_return, eps_len = 0, 0
        obs= self.train_env.reset()
        for t in pbar:
            if (t + 1) < update_after:
                obs["action"] = action_spec.sample()
            else:
                with torch.no_grad():
                    obs = self.actor(obs)
            obs = self.train_env.step(obs)
            obs["next"]["reward"] *= 0 # remove reward
            self.replay_buffer.extend(obs)
            obs = obs["next"]
            
            # end of trajectory handeling
            if obs["terminated"].all() or obs["step_count"] >= max_eps_steps:
                eps_return = obs["episode_reward"].cpu().numpy().mean()
                eps_len = obs["step_count"].cpu().numpy().mean()
                if self.logger is not None:
                    self.logger.log_scalar("train_eps_return", obs["episode_reward"], t)
                    self.logger.log_scalar("train_eps_len", obs["step_count"], t)

                # start new episode
                obs = self.train_env.reset()

            # train reward
            if (t + 1) > update_after and (t - update_after + 1) % reward_update_every == 0:
                reward_stats_epoch = self.train_reward_epoch(t)

            # train policy
            if (t + 1) > update_after and (t - update_after + 1) % policy_update_every == 0:
                policy_stats_epoch = self.train_policy_epoch(t)

            # end of epoch handeling
            if (t + 1) > update_after and (t - update_after + 1) % steps_per_epoch == 0:
                epoch = (t - update_after + 1) // steps_per_epoch

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
                
                desc_str = f"epoch: {epoch}/{epochs}, step: {t}, train_eps_return: {eps_return:.2f}, train_eps_len: {eps_len}"
                pbar.set_description(desc_str)

                # checkpoint handling
                if save_steps not in [None, -1] and (epoch + 1) % save_steps == 0:
                    self.save()
        
        pbar.close()
        self.close_envs()
        self.save()