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
from cleanil.rl.actor import TanhNormalActor, sample_actor, compute_action_likelihood
from cleanil.dynamics.ensemble_dynamics import EnsembleDynamics
from cleanil.il.reward import RewardModel, compute_grad_penalty
from cleanil.utils import (
    freeze_model_parameters, 
    compute_linear_scale, 
    concat_tensordict_on_shared_keys, 
    compute_parameter_l2,
)

from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.data import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import Logger


@dataclass
class OMLIRLConfig(BaseTrainerConfig):
    # data args
    expert_dataset: str = "halfcheetah-expert-v2"
    transition_dataset: str = "halfcheetah-medium-replay-v2"
    pretrained_model_path: str = "../../exp/dynamics/ensemble/halfcheetah-medium-v2/model.p"
    num_expert_trajs: int = 10
    transition_data_size: int = 100000
    upsample: bool = False

    # reward args
    max_reward: float = 10.
    reward_state_only: bool = False
    reward_use_done: bool = False
    reward_grad_target: float = 1.
    reward_grad_penalty: float = 1.
    reward_l2_penalty: float = 1.e-5
    reward_rollout_batch_size: int = 64
    reward_rollout_steps: int = 100
    lr_reward: float = 3.e-4
    update_reward_every: int = 1000
    reward_train_steps: int = 1

    # model args
    lam: float = 1.
    lam_target: float = 1.5
    tune_lam: bool = True
    lr_lam: float = 0.01

    # rollout args
    rollout_batch_size: int = 5000
    rollout_min_steps: int = 40
    rollout_max_steps: int = 40
    rollout_min_epoch: int = 50
    rollout_max_epoch: int = 200
    rollout_max_obs: float = 30.
    sample_model_every: int = 250
    model_retain_epochs: int = 5

    # rl args
    closed_form_terminal: bool = False
    hidden_dims: list = (256, 256)
    activation: str = "silu"
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    tune_alpha: float = True
    batch_size: int = 256
    real_ratio: float = 0.5
    policy_train_steps: int = 1
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
    closed_form_terminal: bool,
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
            rwd, v_next, done, gamma, 
            closed_form_terminal=closed_form_terminal
        )

    q1, q2 = critic(obs, act)
    q1_loss = nn.SmoothL1Loss()(q1, q_target)
    q2_loss = nn.SmoothL1Loss()(q2, q_target)
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
    closed_form_terminal: bool,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    optimizer.zero_grad()
    loss, stats = compute_critic_loss(
        batch, critic, critic_target, actor, gamma, alpha, closed_form_terminal
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

@torch.compile
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
    next_obs = obs["observation"] + delta
    terminated = obs["terminated"].clone()
    if termination_fn is not None:
        terminated = termination_fn(next_obs)
        terminated = torch.cat([obs["terminated"], terminated], dim=-1).any(-1, keepdim=True)

    obs["next"] = TensorDict(
        {"observation": next_obs, "terminated": terminated}, 
        batch_size=len(next_obs),
    )
    return obs

def compute_reward_loss(
    real_batch: TensorDict,
    fake_batch: TensorDict,
    reward: RewardModel,
    gamma: float,
    grad_target: float,
    grad_penalty: float,
    l2_penalty: float,
):
    """Trajectories have dimensions [batch_size, seq_len, ...]"""
    real_rwd_inputs = reward.compute_inputs(
        real_batch["observation"], real_batch["action"], real_batch["terminated"].float()
    )
    fake_rwd_inputs = reward.compute_inputs(
        fake_batch["observation"], fake_batch["action"], fake_batch["terminated"].float()
    )
    real_rwd = reward.forward_on_inputs(real_rwd_inputs)
    fake_rwd = reward.forward_on_inputs(fake_rwd_inputs)
    
    seq_len = real_rwd_inputs.shape[1]
    gamma_seq = gamma ** torch.arange(seq_len).view(1, -1, 1)
    real_return = torch.sum(gamma_seq * real_rwd, dim=1) / seq_len
    fake_return = torch.sum(gamma_seq * fake_rwd, dim=1) / seq_len

    reward_loss = -(real_return.mean() - fake_return.mean())
    gp, g_norm = compute_grad_penalty(
        real_rwd_inputs.flatten(0, 1), 
        fake_rwd_inputs.flatten(0, 1), 
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
    gamma: float,
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
        gamma,
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
    """Offline maximum likelihood inverse reinforcement learning

    Using MOPO with automatic ensemble penalty tuning as the RL solver [2, 3].

    [1] When Demonstrations Meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning, Zeng et al, 2023 (https://arxiv.org/abs/2302.07457)
    [2] MOPO: Model-based Offline Policy Optimization, Yu et al, 2020 (https://arxiv.org/abs/2005.13239)
    [3] Revisiting Design Choices in Offline Model-Based Reinforcement Learning, Lu et al, 2022 (https://arxiv.org/abs/2110.04135)
    """
    def __init__(
        self, 
        config: OMLIRLConfig,
        actor: TanhNormalActor,
        critic: DoubleQNetwork,
        dynamics: EnsembleDynamics,
        reward: RewardModel,
        expert_buffer: ReplayBuffer,
        transition_buffer: ReplayBuffer,
        eval_env: Optional[TransformedEnv] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        termination_fn: Callable | None = None,
        logger: Optional[Logger] = None,
        device: torch.device = "cpu",
    ):
        super().__init__(
            config, None, eval_env, replay_buffer, logger, device
        )
        self.config = config
        self.termination_fn = termination_fn

        self.expert_buffer = expert_buffer
        self.transition_buffer = transition_buffer

        self.reward = reward
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
            "reward": torch.optim.Adam(
                self.reward.parameters(), lr=config.lr_reward
            ),
        }

        # store modules for checkpointing
        self._modules["actor"] = self.actor
        self._modules["critic"] = self.critic
        self._modules["critic_target"] = self.critic_target
        self._modules["log_alpha"] = self.log_alpha
        self._modules["dynamics"] = self.dynamics
        self._modules["log_lam"] = self.log_lam
        self._modules["reward"] = self.reward
    
    def take_policy_gradient_step(self, batch: TensorDict, global_step: int):
        gamma = self.config.gamma
        alpha = self.alpha
        tune_alpha = self.config.tune_alpha
        closed_form_terminal = self.config.closed_form_terminal
        grad_clip = self.config.grad_clip

        # replace data with learned reward
        with torch.no_grad():
            batch["reward"] = self.reward.forward(
                batch["observation"], batch["action"], batch["terminated"].float()
            )
            ensemble_pen = compute_ensemble_penalty(self.dynamics, batch)
            batch["reward"] -= self.lam * ensemble_pen

        critic_stats = update_critic(
            batch,
            self.critic,
            self.critic_target,
            self.actor,
            gamma,
            alpha,
            closed_form_terminal,
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
    
    def take_reward_gradient_step(self, real_batch: TensorDict, fake_batch: TensorDict, global_step: int):
        gamma = self.config.gamma
        grad_target = self.config.reward_grad_target
        grad_penalty = self.config.reward_grad_penalty
        l2_penalty = self.config.reward_l2_penalty
        grad_clip = self.config.grad_clip

        reward_stats = update_reward(
            real_batch,
            fake_batch,
            self.reward,
            gamma,
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
        batch_size = self.config.reward_rollout_batch_size
        traj_len = self.config.reward_rollout_steps

        reward_stats_epoch = []
        for _ in range(train_steps):
            real_batch = self.expert_buffer.sample(batch_size * traj_len).reshape(-1, traj_len)
            fake_batch = self.sample_imagined_data(
                batch_size, 
                traj_len, 
                obs=real_batch["observation"][:, 0], 
                terminated_early=False,
                add_penalty=False,
                save_to_buffer=False,
                return_data=True,
            )
            reward_stats = self.take_reward_gradient_step(real_batch, fake_batch, global_step)
            reward_stats_epoch.append(reward_stats)

        reward_stats_epoch = tree_map(lambda *x: np.stack(x).mean(0), *reward_stats_epoch)
        return reward_stats_epoch
    
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
    
    def sample_imagined_data(
        self, 
        rollout_batch_size: int, 
        rollout_steps: int, 
        global_step: int | None = None, 
        obs: torch.Tensor | None = None,
        terminated_early: bool = True,
        add_penalty: bool = True,
        save_to_buffer: bool = True,
        return_data: bool = False,
    ):
        rollout_max_obs = self.config.rollout_max_obs

        if obs is None:
            obs = self.transition_buffer.sample(rollout_batch_size)["observation"]
        
        obs = TensorDict(
            {
                "observation": obs.clone(), 
                "terminated": torch.zeros(rollout_batch_size, 1, device=obs.device).bool(),
            },
            batch_size=rollout_batch_size,
        )

        data = []
        for t in range(rollout_steps):
            with torch.no_grad():
                obs = self.actor(obs)
            obs = step_dynamics(self.dynamics, obs, termination_fn=self.termination_fn)

            if add_penalty:
                ensemble_pen = compute_ensemble_penalty(self.dynamics, obs)
                self.update_lam(ensemble_pen)
                if self.logger is not None:
                    self.logger.log_scalar(f"dynamics/ensemble_pen", ensemble_pen.mean().cpu().numpy(), global_step)

            if save_to_buffer:
                self.replay_buffer.extend(obs)
            
            if return_data:
                data.append(obs.clone())
            
            obs = obs["next"]

            if terminated_early:
                # obs = obs[obs["terminated"].flatten() == False]
                obs = obs[torch.all(obs["observation"].abs() < rollout_max_obs, dim=-1)]

            if len(obs) == 0:
                break

        if return_data:
            data = torch.stack(data, dim=1)
            return data
    
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
        rollout_batch_size = self.config.rollout_batch_size
        sample_model_every = config.sample_model_every
        update_reward_every = config.update_reward_every
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
                self.sample_imagined_data(rollout_batch_size, model_rollout_steps, t)
                if self.logger is not None:
                    self.logger.log_scalar("dynamics/model_rollout_steps", model_rollout_steps, t)
                    self.logger.log_scalar("dynamics/model_buffer_size", len(self.replay_buffer), t)
                    self.logger.log_scalar("dynamics/lam", self.lam[0].detach().cpu().numpy(), t)

            # train policy
            policy_stats_epoch = self.train_policy_epoch(t)

            # train reward
            if (t + 1) % update_reward_every == 0:
                reward_stats_epoch = self.train_reward_epoch(t)

            # end of epoch handeling
            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                # evaluate episodes
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    eval_return = 0.
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
                # if save_steps not in [None, -1] and (epoch + 1) % save_steps == 0:
                    # self.save()
        
        pbar.close()
        self.close_envs()
        self.save()