import time
import copy
from dataclasses import dataclass
from typing import Optional, Callable
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from optree import tree_map
from cleanil.base_trainer import BaseTrainer, BaseTrainerConfig
from cleanil.rl.critic import DoubleQNetwork, compute_q_target, update_critic_target
from cleanil.rl.actor import TanhNormalActor, sample_actor
from cleanil.dynamics.ensemble_dynamics import EnsembleDynamics
from cleanil.utils import (
    freeze_model_parameters,
    compute_linear_scale, 
    concat_tensordict_on_shared_keys,
)

from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import Logger


@dataclass
class MOPOConfig(BaseTrainerConfig):
    dataset: str = "halfcheetah-medium-expert-v2"
    pretrained_model_path: str = "../../exp/dynamics/ensemble/halfcheetah-medium-expert-v2/model.p"
    transition_data_size: int = 2000000

    # model args
    lam: float = 1.
    lam_target: float = 1.
    tune_lam: bool = True
    lr_lam: float = 0.01

    # rollout args
    rollout_batch_size: int = 5000
    rollout_min_steps: int = 40
    rollout_max_steps: int = 40
    rollout_min_epoch: int = 50
    rollout_max_epoch: int = 200
    sample_model_every: int = 250
    model_retain_epochs: int = 5

    # rl args
    hidden_dims: list = (256, 256)
    activation: str = "silu"
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    tune_alpha: float = True
    batch_size: int = 256
    real_ratio: float = 0.5
    policy_train_steps: int = 10
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    grad_clip: float = 1000.

    # train args
    buffer_size: int = 1e6
    max_eps_steps: int = 1000
    epochs: int = 1000
    steps_per_epoch: int = 1000
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
    rwd = batch["next"]["reward"]
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

@torch.no_grad()
def compute_ensemble_penalty(
    dynamics: EnsembleDynamics,
    obs: TensorDict,
):
    """Compute ensemble standard deviation penalty"""
    dist = dynamics.get_dist(obs["observation"], obs["action"])
    mean = dist.mean
    variance = dist.variance
    mean_of_vars = torch.mean(variance, dim=-2)
    var_of_means = torch.var(mean, dim=-2)
    std = (mean_of_vars + var_of_means).sqrt()
    pen = torch.mean(std, dim=-1, keepdim=True)
    return pen

@torch.no_grad()
def step_dynamics(
    dynamics: EnsembleDynamics,
    obs: TensorDict,
    termination_fn: Callable | None = None,
):
    delta = dynamics.sample_dist(obs["observation"], obs["action"])
    delta, rwd = delta[..., :-1], delta[..., -1:]
    next_obs = obs["observation"] + delta
    terminated = torch.zeros(len(next_obs), 1, device=next_obs.device).bool()
    if termination_fn is not None:
        terminated = termination_fn(next_obs)

    obs["next"] = TensorDict(
        {"observation": next_obs, "reward": rwd, "delta": delta, "terminated": terminated}, 
        batch_size=len(next_obs),
    )
    return obs


class Trainer(BaseTrainer):
    """Model based offline policy optimization

    With automatic uncertainty penalty tuning [2].
    
    [1] MOPO: Model-based Offline Policy Optimization, Yu et al, 2020 (https://arxiv.org/abs/2005.13239)
    [2] Revisiting Design Choices in Offline Model-Based Reinforcement Learning, Lu et al, 2022 (https://arxiv.org/abs/2110.04135)
    """
    def __init__(
        self, 
        config: MOPOConfig,
        actor: TanhNormalActor,
        critic: DoubleQNetwork,
        dynamics: EnsembleDynamics,
        transition_buffer: ReplayBuffer,
        eval_env: Optional[TransformedEnv] = None,
        termination_fn: Callable | None = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        logger: Optional[Logger] = None,
        device: torch.device = "cpu",
    ):
        super().__init__(
            config, None, eval_env, replay_buffer, logger, device
        )
        self.config = config
        self.termination_fn = termination_fn

        self.dynamics = dynamics
        self.log_lam = nn.Parameter(
            np.log(config.lam) * torch.ones(1, device=device), 
            requires_grad=config.tune_lam,
        )
        self.lam = self.log_lam.data.exp()
        self.lam_target = config.lam_target

        self.actor = actor
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = nn.Parameter(
            np.log(config.alpha) * torch.ones(1, device=device), 
            requires_grad=config.tune_alpha,
        )
        self.alpha = self.log_alpha.data.exp()
        self.alpha_target = -torch.prod(torch.tensor(eval_env.action_spec.shape, device=device))

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
            "lam": torch.optim.Adam(
                [self.log_lam], lr=config.lr_lam
            ),
        }

        self.transition_buffer = transition_buffer

        # store modules for checkpointing
        self._modules["actor"] = self.actor
        self._modules["critic"] = self.critic
        self._modules["critic_target"] = self.critic_target
        self._modules["log_alpha"] = self.log_alpha
        self._modules["dynamics"] = self.dynamics
        self._modules["log_lam"] = self.log_lam

    def load_state_dict(self, state_dict: dict):
        self.dynamics.load_state_dict(state_dict["dynamics"])
        self.log_lam.data = state_dict["log_lam"].data

        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.log_alpha.data = state_dict["log_alpha"].data
    
    def take_policy_gradient_step(self, batch: TensorDict, global_step: int):
        gamma = self.config.gamma
        alpha = self.alpha
        tune_alpha = self.config.tune_alpha
        grad_clip = self.config.grad_clip

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

        real_batch_size = batch_size
        fake_batch_size = 0
        if real_ratio > 0 and len(self.replay_buffer) >= batch_size:
            real_batch_size = int(batch_size * real_ratio)
            fake_batch_size = batch_size - real_batch_size

        policy_stats_epoch = []
        for _ in range(train_steps):
            batch = self.transition_buffer.sample(real_batch_size)
            if fake_batch_size > 0:
                fake_batch = self.replay_buffer.sample(fake_batch_size)
                batch = concat_tensordict_on_shared_keys(fake_batch, batch)

            policy_stats = self.take_policy_gradient_step(batch, global_step)
            policy_stats_epoch.append(policy_stats)

        policy_stats_epoch = tree_map(lambda *x: np.stack(x).mean(0), *policy_stats_epoch)
        return policy_stats_epoch
    
    def compute_model_rollout_steps(self, epoch: int):
        rollout_steps = compute_linear_scale(
            self.config.rollout_min_steps, 
            self.config.rollout_max_steps, 
            self.config.rollout_min_epoch, 
            self.config.rollout_max_epoch,
            epoch
        )
        return int(rollout_steps)
    
    def update_lam(self, ensemble_pen):
        self.optimizers["lam"].zero_grad()
        lam = self.log_lam.exp()
        lam_loss = torch.mean(self.log_lam.exp() * (lam * ensemble_pen - self.lam_target).detach())
        
        if self.config.tune_lam:
            lam_loss.backward()
            self.optimizers["lam"].step()
            with torch.no_grad():
                self.lam = self.log_lam.data.exp()
    
    def sample_imagined_data(self, rollout_steps: int, global_step: int):
        rollout_batch_size = self.config.rollout_batch_size
        
        obs = self.transition_buffer.sample(rollout_batch_size).clone()
        obs = TensorDict(
            {
                "observation": obs["observation"], 
                "action": obs["action"],
            },
            batch_size=rollout_batch_size,
        )
        for t in range(rollout_steps):
            with torch.no_grad():
                obs = self.actor(obs)
            obs = step_dynamics(self.dynamics, obs, termination_fn=self.termination_fn)
            ensemble_pen = compute_ensemble_penalty(self.dynamics, obs)
            obs["next"]["reward"] -= self.lam * ensemble_pen
            self.update_lam(ensemble_pen)
            self.replay_buffer.extend(obs)
            
            obs = obs["next"]

            # terminate early
            obs = obs[obs["terminated"].flatten() == False]

            if self.logger is not None:
                self.logger.log_scalar(f"dynamics/ensemble_pen", ensemble_pen.mean().cpu().numpy(), global_step)
            
            if len(obs) == 0:
                break

    def train(self):
        config = self.config
        epochs = config.epochs
        steps_per_epoch = config.steps_per_epoch
        sample_model_every = config.sample_model_every
        max_eps_steps = config.max_eps_steps
        num_eval_eps = config.num_eval_eps
        save_steps = config.save_steps

        total_steps = epochs * steps_per_epoch
        start_time = time.time()
        
        epoch = 0
        pbar = tqdm(range(total_steps))
        for t in pbar:
            # train model
            if t == 0 or (t + 1) % sample_model_every == 0:
                model_rollout_steps = self.compute_model_rollout_steps(epoch)
                self.sample_imagined_data(model_rollout_steps, t)
                if self.logger is not None:
                    self.logger.log_scalar("dynamics/model_rollout_steps", model_rollout_steps, t)
                    self.logger.log_scalar("dynamics/model_buffer_size", len(self.replay_buffer), t)
                    self.logger.log_scalar("dynamics/lam", self.lam[0].detach().cpu().numpy(), t)

            # train policy
            policy_stats_epoch = self.train_policy_epoch(t)

            # end of epoch handeling
            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                # evaluate episodes
                eval_return = 0.
                if num_eval_eps > 0:
                    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
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

                # checkpoint handling
                if save_steps not in [None, -1] and (epoch + 1) % save_steps == 0:
                    self.save()
        
        pbar.close()
        self.close_envs()
        self.save()