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
from cleanil.rl.critic import DoubleQNetwork, update_critic_target
from cleanil.rl.actor import TanhNormalActor, sample_actor, compute_action_likelihood
from cleanil.dynamics.ensemble_dynamics import (
    EnsembleDynamics,
    format_samples_for_training, 
    get_random_index,
)
from cleanil.il.reward import compute_grad_penalty
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
class RMIRLImplicitConfig(BaseTrainerConfig):
    # data args
    expert_dataset: str = "halfcheetah-expert-v2"
    transition_dataset: str = "halfcheetah-medium-replay-v2"
    pretrained_model_path: str = "../../exp/dynamics/ensemble/halfcheetah-medium-replay-v2"
    num_expert_trajs: int = 10
    transition_data_size: int = 100000
    train_ratio: float = 0.9
    upsample: bool = False

    # reward args
    q_max: float = 1000.
    use_double: bool = True
    use_done: bool = False
    ibc_penalty: float = 1.
    td_penalty: float = 0.5
    grad_target: float = 600.
    grad_penalty: float = 10.
    decouple_loss: bool = True

    # model train args
    model_eval_ratio: float = 0.2
    obs_penalty: float = 1.
    adv_penalty: float = 0.05
    adv_rollout_steps: int = 20
    adv_clip_max: float = 10.
    norm_advantage: bool = True
    lr_model: float = 1.e-4
    model_train_batch_size: int = 256
    model_train_steps: int = 100
    update_model_every: int = 1000

    # rollout args
    rollout_expert_ratio: float = 0.5
    rollout_batch_size: int = 5000
    rollout_min_steps: int = 20
    rollout_max_steps: int = 20
    rollout_min_epoch: int = 50
    rollout_max_epoch: int = 200
    rollout_max_obs: float = 30.
    sample_model_every: int = 250
    model_retain_epochs: int = 5

    # rl args
    hidden_dims: list = (256, 256)
    activation: str = "silu"
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.1
    tune_alpha: float = False
    batch_size: int = 256
    policy_train_steps: int = 1
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_critic_min: float = 1e-5
    warmup_epochs: int = 300
    grad_clip: float = 1000.

    # train args
    buffer_size: int = 1e6
    max_eps_steps: int = 1000
    epochs: int = 1000
    steps_per_epoch: int = 1000
    num_eval_eps: int = 1
    eval_steps: int = 1000


class CriticWrapper(nn.Module):
    def __init__(self, critic: DoubleQNetwork, use_double: bool, use_done: bool, q_max: float):
        super().__init__()
        # init critic weights
        critic.q1[-1].weight.data *= 0.
        critic.q1[-1].bias.data *= 0.
        critic.q2[-1].weight.data *= 0.
        critic.q2[-1].bias.data *= 0.

        self.critic = critic

        self.use_double = use_double
        self.use_done = use_done
        self.q_max = q_max

    def compute_inputs(self, obs, act, done):
        inputs = torch.cat([obs, act], dim=-1)
        if self.use_done:
            inputs = torch.cat([(1 - done) * inputs, done], dim=-1)
        return inputs
    
    def forward(self, obs, act, done, clip=False):
        critic_inputs = self.compute_inputs(obs, act, done)
        
        if self.use_double:
            q1 = self.critic.q1(critic_inputs)
            q2 = self.critic.q2(critic_inputs)
        else:
            q1 = self.critic.q1(critic_inputs)
            q2 = q1

        if clip:
            q1 = q1.clip(-self.q_max, self.q_max)
            q2 = q2.clip(-self.q_max, self.q_max)
        return q1, q2
    
    def compute_grad_penalty(self, obs, act, done, grad_target: float):
        critic_inputs = self.compute_inputs(obs, act, 0. * done)

        if self.use_double:
            gp1, g_norm1 = compute_grad_penalty(
                critic_inputs, critic_inputs, lambda x: self.critic.q1(x), grad_target
            )
            gp2, g_norm2 = compute_grad_penalty(
                critic_inputs, critic_inputs, lambda x: self.critic.q2(x), grad_target
            )
            gp = (gp1 + gp2) / 2
            g_norm = (g_norm1 + g_norm2) / 2
        else:
            gp, g_norm = compute_grad_penalty(
                critic_inputs, critic_inputs, lambda x: self.critic.q1(x), grad_target
            )
        return gp, g_norm
    

def compute_critic_loss(
    expert_batch: TensorDict, 
    transition_batch: TensorDict,
    td_batch: TensorDict,
    critic: CriticWrapper, 
    critic_target: CriticWrapper, 
    actor: TanhNormalActor, 
    gamma: float, 
    alpha: float | torch.Tensor,
    ibc_penalty: float,
    td_penalty: float,
    grad_target: float,
    grad_penalty: float,
) -> torch.Tensor:
    obs_expert = expert_batch["observation"]
    act_expert = expert_batch["action"]
    done_expert = expert_batch["terminated"].float()

    obs_transition = transition_batch["observation"]
    act_transition = transition_batch["action"]
    done_transition = transition_batch["terminated"].float()

    obs = td_batch["observation"]
    act = td_batch["action"]
    done = td_batch["terminated"].float()
    next_obs = td_batch["next"]["observation"]
    next_done = td_batch["next"]["terminated"].float()

    with torch.no_grad():
        # sample fake action expert
        fake_act_expert, _ = sample_actor(obs_expert, actor)

        # compute value target
        next_act, logp = sample_actor(next_obs, actor)
        q1_next, q2_next = critic_target.forward(next_obs, next_act, next_done, clip=True)
        q_next = torch.min(q1_next, q2_next)
        v_next = q_next - alpha * logp
        q_target = gamma * v_next

    q1_expert, q2_expert = critic.forward(obs_expert, act_expert, done_expert, clip=False)
    q1_fake, q2_fake = critic.forward(obs_expert, fake_act_expert, done_expert, clip=False)
    q1_transition, q2_transition = critic.forward(obs_transition, act_transition, done_transition, clip=False)

    # compute ibc loss
    ibc_loss_1 = -(q1_expert.mean() - q1_fake.mean())
    ibc_loss_2 = -(q2_expert.mean() - q2_fake.mean())
    ibc_loss = (ibc_loss_1 + ibc_loss_2) / 2

    # compute transition contrastive loss
    cd_loss_1 = -(q1_expert.mean() - q1_transition.mean())
    cd_loss_2 = -(q2_expert.mean() - q2_transition.mean())
    cd_loss = (cd_loss_1 + cd_loss_2) / 2

    # compute td loss
    q1, q2 = critic.forward(obs, act, done, clip=False)
    td_loss_1 = torch.pow(q1 - q_target, 2).mean()
    td_loss_2 = torch.pow(q2 - q_target, 2).mean()
    td_loss = (td_loss_1 + td_loss_2) / 2

    total_loss = cd_loss + ibc_penalty * ibc_loss + td_penalty * td_loss

    with torch.no_grad():
        q_expert = torch.cat([q1_expert, q2_expert], dim=0)
        q_fake = torch.cat([q1_fake, q2_fake], dim=0)
        q_transition = torch.cat([q1_transition, q2_transition], dim=0)

    stats = {
        "critic_loss": total_loss.detach().cpu().numpy(),
        "critic_ibc_loss": ibc_loss.detach().cpu().numpy(),
        "critic_cd_loss": cd_loss.detach().cpu().numpy(),
        "critic_td_loss": td_loss.detach().cpu().numpy(),
        "critic_expert_value": q_expert.mean().detach().cpu().numpy(),
        "critic_transition_value": q_transition.mean().detach().cpu().numpy(),
        "critic_fake_value": q_fake.mean().detach().cpu().numpy(),
    }

    # compute grad penalty
    if grad_penalty > 0:
        _done_expert = torch.zeros_like(done_expert, device=done_expert.device)
        _done_transition = torch.zeros_like(done_transition, device=done_transition.device)
        _done = torch.cat([_done_expert, _done_transition], dim=0)
        _obs = torch.cat([obs_expert, obs_transition], dim=0)
        _act = torch.cat([act_expert, act_transition], dim=0)
        gp, g_norm = critic.compute_grad_penalty(_obs, _act, _done, grad_target)

        total_loss += grad_penalty * gp

        stats.update({
            "critic_grad_pen": gp.detach().cpu().numpy(),
            "critic_grad_norm": g_norm.detach().cpu().numpy().mean(),
        })
    return total_loss, stats

@torch.compile
def update_critic(
    expert_batch: TensorDict, 
    transition_batch: TensorDict, 
    td_batch: TensorDict,
    critic: CriticWrapper, 
    critic_target: CriticWrapper, 
    actor: TanhNormalActor, 
    gamma: float, 
    alpha: float | torch.Tensor,
    ibc_penalty: float,
    td_penalty: float,
    grad_target: float,
    grad_penalty: float,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    optimizer.zero_grad()
    loss, stats = compute_critic_loss(
        expert_batch, 
        transition_batch, 
        td_batch,
        critic, 
        critic_target, 
        actor, 
        gamma, 
        alpha, 
        ibc_penalty,
        td_penalty,
        grad_target,
        grad_penalty,
    )
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optimizer.step()
    return stats

def compute_actor_loss(
    td_batch: TensorDict,
    critic: CriticWrapper,
    actor: TanhNormalActor,
    alpha: torch.Tensor,
):  
    obs = td_batch["observation"]
    done = torch.zeros(len(obs), 1, device=obs.device) # dummy done

    act, logp = sample_actor(obs, actor)
    q1, q2 = critic.forward(obs, act, done, clip=False)
    q = torch.min(q1, q2)
    actor_loss = torch.mean(alpha * logp - q)

    stats = {
        "actor_loss": actor_loss.detach().cpu().numpy(),
        "alpha": alpha[0].detach().cpu().numpy(),
    }
    return actor_loss, logp, stats

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
    td_batch: TensorDict,
    critic: CriticWrapper,
    actor: TanhNormalActor,
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
        td_batch, critic, actor, alpha
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

def sample_buffer(
    expert_buffer: ReplayBuffer, 
    transition_buffer: ReplayBuffer, 
    batch_size: int, 
    rollout_expert_ratio: float,
):
    batch_size_expert = int(batch_size * rollout_expert_ratio)
    batch_size_transition = batch_size - batch_size_expert

    obs_expert = expert_buffer.sample(batch_size_expert)
    obs_transition = transition_buffer.sample(batch_size_transition)
    obs = torch.cat([obs_expert, obs_transition], dim=0)
    
    device = obs["observation"].device
    delta_dim = obs["observation"].shape[-1]
    obs = TensorDict(
        {
            "observation": obs["observation"],
            "delta": torch.zeros(batch_size, delta_dim, dtype=torch.float32, device=device),
            "terminated": obs["terminated"],
        },
        batch_size=batch_size,
    )
    return obs

@torch.no_grad()
def step_dynamics(
    dynamics: EnsembleDynamics,
    obs: TensorDict,
    termination_fn: Callable | None = None,
):
    delta = dynamics.sample_dist(obs["observation"], obs["action"])
    next_obs = obs["observation"].clone() + delta
    terminated = obs["terminated"].clone()
    if termination_fn is not None:
        terminated = termination_fn(next_obs)
        terminated = torch.cat([obs["terminated"], terminated], dim=-1).any(-1, keepdim=True)

    obs["next"] = TensorDict(
        {"observation": next_obs, "delta": delta, "terminated": terminated},  
        batch_size=len(next_obs),
    )
    return obs

def compute_dynamics_adversarial_loss(
    dynamics: EnsembleDynamics,
    critic: CriticWrapper,
    actor: TanhNormalActor,
    step_dict: TensorDict,
    gamma: float, 
    alpha: float | torch.Tensor,
    norm_advantage: bool,
    adv_clip_max: float,
):
    obs = step_dict["observation"]
    done = step_dict["terminated"].float()
    act = step_dict["action"]
    next_obs = step_dict["next"]["observation"]
    next_done = step_dict["next"]["terminated"].float()
    target = step_dict["next"]["delta"]
    obs_dim = obs.shape[-1]
    
    # compute advantage
    with torch.no_grad():
        q1, q2 = critic.forward(obs, act, done, clip=True)
        q = torch.min(q1, q2)

        # compute next value
        next_act, logp = sample_actor(next_obs, actor)
        q1_next, q2_next = critic.forward(next_obs, next_act, next_done, clip=True)
        q_next = torch.min(q1_next, q2_next)
        v_next = q_next - alpha * logp
        advantage = gamma * v_next - q.data
        
        if norm_advantage:
            advantage_norm = (advantage - advantage.mean(0)) / (advantage.std(0) + 1e-6)
        else:
            advantage_norm = advantage
        advantage_norm = advantage_norm.clip(-adv_clip_max, adv_clip_max)
    
    # compute ensemble mixture log likelihood
    logp = dynamics.compute_mixture_log_prob(obs, act, target)
    adv_loss = torch.mean(advantage_norm * logp) / (obs_dim + 1)
    
    stats = {
        "adv_v_next_mean": v_next.detach().cpu().mean().item(),
        "adv_mean": advantage.detach().cpu().mean().item(),
        "adv_std": advantage.detach().std().data.item(),
        "adv_logp_mean": logp.detach().mean().data.item(),
        "adv_logp_std": logp.detach().std().data.item(),
    }
    return adv_loss, stats


class Trainer(BaseTrainer):
    """Robust model based inverse reinforcement learning with implicit reward

    This modifies RMIRL [2] to use implicit reward parameterized by Q function as proposed in IQ-Learn [3].
    The resulting critic loss function is similar to RECOIL [4].
    The dynamics model is still trained adversarially.

    [1] Implicit vs. Explicit Offline Inverse Reinforcement Learning: A Credit Assignment Perspective, Wei et al, 2025 (https://openreview.net/forum?id=X3YuA7z2iX)
    [2] A Bayesian Approach to Robust Inverse Reinforcement Learning, Wei et al, 2023 (https://arxiv.org/abs/2309.08571)
    [3] IQ-Learn: Inverse soft-Q Learning for Imitation, Garg et al, 2021 (https://arxiv.org/abs/2106.12142)
    [4] Dual RL: Unification and New Methods for Reinforcement and Imitation Learning, Sikchi et al, 2023 (https://arxiv.org/abs/2302.08560)
    """
    def __init__(
        self, 
        config: RMIRLImplicitConfig,
        actor: TanhNormalActor,
        critic: DoubleQNetwork,
        dynamics: EnsembleDynamics,
        expert_buffer: ReplayBuffer,
        transition_buffer: ReplayBuffer,
        eval_buffer: ReplayBuffer,
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
        self.eval_buffer = eval_buffer

        self.dynamics = dynamics

        critic = CriticWrapper(critic, config.use_double, config.use_done, config.q_max)

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
            "dynamics": torch.optim.Adam(
                self.dynamics.parameters(), lr=config.lr_model
            ),
        }

        self.warmup_steps = config.steps_per_epoch * config.warmup_epochs
        self.schedulers = {
            "critic": torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers["critic"], self.warmup_steps, config.lr_critic_min,
            ),
        }

        # store modules for checkpointing
        self._modules["actor"] = self.actor
        self._modules["critic"] = self.critic
        self._modules["critic_target"] = self.critic_target
        self._modules["log_alpha"] = self.log_alpha
        self._modules["dynamics"] = self.dynamics
    
    def take_policy_gradient_step(
        self, expert_batch: TensorDict, transition_batch: TensorDict, td_batch: TensorDict, global_step: int
    ):
        gamma = self.config.gamma
        alpha = self.alpha
        ibc_penalty = self.config.ibc_penalty
        td_penalty = self.config.td_penalty
        grad_target = self.config.grad_target
        grad_penalty = self.config.grad_penalty
        tune_alpha = self.config.tune_alpha
        grad_clip = self.config.grad_clip

        critic_stats = update_critic(
            expert_batch,
            transition_batch,
            td_batch,
            self.critic,
            self.critic_target,
            self.actor,
            gamma,
            alpha,
            ibc_penalty,
            td_penalty,
            grad_target,
            grad_penalty,
            self.optimizers["critic"],
            grad_clip,
        )

        actor_stats = update_actor(
            td_batch,
            self.critic,
            self.actor,
            alpha,
            self.log_alpha,
            self.alpha_target,
            tune_alpha,
            self.optimizers["actor"],
            self.optimizers["alpha"],
        )

        update_critic_target(self.critic, self.critic_target, self.config.polyak)
        self.alpha = self.log_alpha.detach().exp()

        if (global_step + 1) <= self.warmup_steps:
            self.schedulers["critic"].step()
        
        metrics = {**critic_stats, **actor_stats}

        if self.logger is not None:
            for metric_name, metric_value in critic_stats.items():
                self.logger.log_scalar(f"critic/{metric_name}", metric_value, global_step)
            for metric_name, metric_value in actor_stats.items():
                self.logger.log_scalar(f"actor/{metric_name}", metric_value, global_step)
        return metrics
    
    def train_policy_epoch(self, global_step: int):
        train_steps = self.config.policy_train_steps
        batch_size = self.config.batch_size
        decouple = self.config.decouple_loss

        policy_stats_epoch = []
        for _ in range(train_steps):
            expert_batch = self.expert_buffer.sample(batch_size // 2)
            transition_batch = self.replay_buffer.sample(batch_size // 2)

            if decouple:
                expert_batch_td = self.expert_buffer.sample(batch_size // 2)
                transition_batch_td = self.replay_buffer.sample(batch_size // 2)
                td_batch = concat_tensordict_on_shared_keys(expert_batch_td, transition_batch_td)
            else:
                td_batch = concat_tensordict_on_shared_keys(expert_batch, transition_batch)
                
            policy_stats = self.take_policy_gradient_step(expert_batch, transition_batch, td_batch, global_step)
            policy_stats_epoch.append(policy_stats)

        policy_stats_epoch = tree_map(lambda *x: np.stack(x).mean(0), *policy_stats_epoch)
        return policy_stats_epoch
    
    def take_dynamics_gradient_step(
        self, obs: TensorDict, sl_inputs: torch.Tensor, sl_targets: torch.Tensor, global_step: int
    ):
        gamma = self.config.gamma
        norm_advantage = self.config.norm_advantage
        adv_clip_max = self.config.adv_clip_max
        obs_penalty = self.config.obs_penalty
        adv_penalty = self.config.adv_penalty
        grad_clip = self.config.grad_clip

        self.optimizers["dynamics"].zero_grad()

        adv_loss, stats = compute_dynamics_adversarial_loss(
            self.dynamics,
            self.critic,
            self.actor,
            obs,
            gamma, 
            self.alpha,
            norm_advantage,
            adv_clip_max,
        )
        sl_loss = self.dynamics.compute_loss(sl_inputs, sl_targets)
        total_loss = adv_penalty * adv_loss + obs_penalty * sl_loss

        total_loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(self.dynamics.parameters(), grad_clip)
        self.optimizers["dynamics"].step()

        stats["sl_loss"] = sl_loss.detach().cpu().numpy()
        if self.logger is not None:
            for metric_name, metric_value in stats.items():
                self.logger.log_scalar(f"dynamics/{metric_name}", metric_value, global_step)
        return stats

    def train_dynamics_epoch(self, global_step: int):
        train_steps = self.config.model_train_steps
        adv_rollout_steps = self.config.adv_rollout_steps
        rollout_max_obs = self.config.rollout_max_obs
        batch_size = self.config.model_train_batch_size
        eval_ratio = self.config.model_eval_ratio
        max_eval_num = 1000
        pred_rwd = False

        # train test split supervised learning data
        num_total = min(int(batch_size * train_steps * (1 + eval_ratio)), len(self.transition_buffer))
        data = self.transition_buffer.sample(num_total)

        train_inputs, train_targets, eval_inputs, eval_targets = format_samples_for_training(
            data, self.dynamics, pred_rwd, eval_ratio, max_eval_num,
        )

        # shuffle train data
        ensemble_dim = self.dynamics.ensemble_dim 
        idx_train = get_random_index(len(train_inputs), ensemble_dim, bootstrap=True)

        rollout_expert_ratio = self.config.rollout_expert_ratio
        dynamics_stats_epoch = []
        counter = 0
        idx_start_sl = 0
        while counter < train_steps:
            obs = sample_buffer(
                self.expert_buffer, self.transition_buffer, batch_size, rollout_expert_ratio
            )
            for t in range(adv_rollout_steps):
                # step dynamics
                with torch.no_grad():
                    obs = self.actor(obs)
                obs = step_dynamics(self.dynamics, obs, termination_fn=self.termination_fn)

                # get supervised batch
                idx_batch = idx_train[idx_start_sl:idx_start_sl+batch_size]
                sl_inputs_batch = train_inputs[idx_batch]
                sl_targets_batch = train_targets[idx_batch]
                
                dynamics_stats = self.take_dynamics_gradient_step(
                    obs, sl_inputs_batch, sl_targets_batch, global_step
                )
                dynamics_stats_epoch.append(dynamics_stats)
                obs = obs["next"]

                # add new initital states
                obs = obs[torch.all(obs["observation"].abs() < rollout_max_obs, dim=-1)]
                obs_init = sample_buffer(
                    self.expert_buffer, self.transition_buffer, batch_size - len(obs), rollout_expert_ratio
                )
                obs = torch.cat([obs, obs_init], dim=0)
                
                counter += 1
                if counter == train_steps:
                    break

                # update sl batch counter
                idx_start_sl += batch_size
                if idx_start_sl + batch_size >= len(train_inputs):
                    idx_start_sl = 0
                    idx_train = get_random_index(len(train_inputs), ensemble_dim, bootstrap=True)

        # evaluate
        eval_stats = self.dynamics.evaluate(eval_inputs, eval_targets)
        if self.logger is not None:
            self.logger.log_scalar(f"dynamics/mae", eval_stats["mae"], global_step)

        dynamics_stats_epoch = tree_map(lambda *x: np.stack(x).mean(0), *dynamics_stats_epoch)
        dynamics_stats_epoch = {**dynamics_stats_epoch, **eval_stats}
        return dynamics_stats_epoch

    def compute_model_rollout_steps(self, epoch: int):
        rollout_steps = compute_linear_scale(
            self.config.rollout_min_steps, 
            self.config.rollout_max_steps, 
            self.config.rollout_min_epoch, 
            self.config.rollout_max_epoch,
            epoch
        )
        return int(rollout_steps)
    
    def sample_imagined_data(
        self, 
        rollout_batch_size: int, 
        rollout_steps: int, 
        obs: torch.Tensor | None = None,
        terminated_early: bool = True,
        save_to_buffer: bool = True,
        return_data: bool = False,
    ):
        rollout_expert_ratio = self.config.rollout_expert_ratio
        rollout_max_obs = self.config.rollout_max_obs

        if obs is None:
            obs = sample_buffer(
                self.expert_buffer, self.transition_buffer, rollout_batch_size, rollout_expert_ratio
            )["observation"]
        
        terminated = self.termination_fn(obs) if self.termination_fn is not None \
            else torch.zeros(rollout_batch_size, 1, device=obs.device).bool()
        obs = TensorDict(
            {
                "observation": obs.clone(), 
                "delta": torch.zeros_like(obs, device=obs.device), 
                "terminated": terminated,
            },
            batch_size=rollout_batch_size,
        )

        data = []
        for t in range(rollout_steps):
            with torch.no_grad():
                obs = self.actor(obs)
            obs = step_dynamics(self.dynamics, obs, termination_fn=self.termination_fn)

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
            eval_batch = self.eval_buffer.sample(num_samples)
            transition_batch = self.transition_buffer.sample(num_samples)
            train_logp, train_mae = eval_func(train_batch)
            eval_logp, eval_mae = eval_func(eval_batch)
            transition_logp, transition_mae = eval_func(transition_batch)

        stats = {
            "train_act_logp": train_logp.mean().detach().cpu().numpy(), 
            "train_act_mae": train_mae.detach().cpu().numpy(),
            "eval_act_logp": eval_logp.mean().detach().cpu().numpy(), 
            "eval_act_mae": eval_mae.detach().cpu().numpy(),
            "transition_act_logp": transition_logp.mean().detach().cpu().numpy(), 
            "transition_act_mae": transition_mae.detach().cpu().numpy(),
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
        update_model_every = config.update_model_every
        model_train_steps = config.model_train_steps
        max_eps_steps = config.max_eps_steps
        num_eval_eps = config.num_eval_eps
        save_steps = config.save_steps

        total_steps = epochs * steps_per_epoch
        start_time = time.time()
        
        epoch = 0
        pbar = tqdm(range(total_steps))
        for t in pbar:
            # sample model
            if t == 0 or (t + 1) % sample_model_every == 0:
                model_rollout_steps = self.compute_model_rollout_steps(epoch)
                self.sample_imagined_data(rollout_batch_size, model_rollout_steps)
                if self.logger is not None:
                    self.logger.log_scalar("dynamics/model_rollout_steps", model_rollout_steps, t)
                    self.logger.log_scalar("dynamics/model_buffer_size", len(self.replay_buffer), t)

            # train policy
            policy_stats_epoch = self.train_policy_epoch(t)

            # train model
            if (t + 1) % update_model_every == 0 and model_train_steps > 0:
                dynamics_stats_epoch = self.train_dynamics_epoch(t)

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
                            self.logger.log_scalar("critic_lr", self.schedulers["critic"].get_last_lr()[0], t)
                    
                    self.eval_action_likelihood(t)
                
                desc_str = f"epoch: {epoch}/{epochs}, step: {t}, eval_eps_return: {eval_return:.2f}"
                pbar.set_description(desc_str)

                # checkpoint handling
                if save_steps not in [None, -1] and (epoch + 1) % save_steps == 0:
                    self.save()
        
        pbar.close()
        self.close_envs()
        self.save()