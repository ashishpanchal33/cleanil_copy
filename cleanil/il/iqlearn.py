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
from cleanil.il.reward import compute_grad_penalty
from cleanil.utils import freeze_model_parameters, compute_parameter_l2

from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import Logger


@dataclass
class IQLearnConfig(BaseTrainerConfig):
    # data args
    dataset: str = "halfcheetah-expert-v2"
    num_expert_trajs: int = 20
    train_ratio: float = 0.9

    # reward args
    actor_loss: str = "awr"
    action_mask_noise: float = 0.1
    grad_target: float = 1.
    grad_penalty: float = 10.
    l2_penalty: float = 1.e-6
    td_penalty: float = 0.5

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
    td_penalty: float,
    grad_target: float,
    grad_penalty: float,
    l2_penalty: float,
) -> torch.Tensor:
    obs = batch["observation"]
    act = batch["action"]
    rwd = torch.zeros(len(obs), 1, device=obs.device)
    done = batch["terminated"].float()
    next_obs = batch["next"]["observation"]

    with torch.no_grad():
        # sample fake action
        fake_act, _ = sample_actor(obs, actor)
        
        # compute value target
        next_act, logp = sample_actor(next_obs, actor)
        q1_next, q2_next = critic_target(next_obs, next_act)
        q_next = torch.min(q1_next, q2_next)
        v_next = q_next - alpha * logp
        q_target = compute_q_target(
            rwd, v_next, done, gamma, closed_form_terminal=False
        )

    q1, q2 = critic(obs, act)
    q1_fake, q2_fake = critic(obs, fake_act)

    # compute contrastive divergence loss
    cd1_loss = -(q1.mean() - q1_fake.mean())
    cd2_loss = -(q2.mean() - q2_fake.mean())
    cd_loss = (cd1_loss + cd2_loss) / 2

    # compute td loss
    td1_loss = torch.pow(q1 - q_target, 2).mean()
    td2_loss = torch.pow(q2 - q_target, 2).mean()
    td_loss = (td1_loss + td2_loss) / 2

    # compute grad penalty
    real_inputs = torch.cat([obs, act], dim=-1)
    fake_inputs = torch.cat([obs, fake_act], dim=-1)
    gp1, g_norm1 = compute_grad_penalty(real_inputs, fake_inputs, lambda x: critic.q1(x), grad_target)
    gp2, g_norm2 = compute_grad_penalty(real_inputs, fake_inputs, lambda x: critic.q2(x), grad_target)
    gp = (gp1 + gp2) / 2
    g_norm = (g_norm1 + g_norm2) / 2

    l2 = compute_parameter_l2(critic)

    total_loss = (1 - td_penalty) * cd_loss + td_penalty * td_loss + grad_penalty * gp + l2_penalty * l2
    
    stats = {
        "critic_loss": total_loss.detach().cpu().numpy(),
        "critic_cd_loss": cd_loss.detach().cpu().numpy(),
        "critic_td_loss": td_loss.detach().cpu().numpy(),
        "critic_grad_pen": gp.detach().cpu().numpy(),
        "critic_grad_norm": g_norm.detach().cpu().numpy().mean(),
        "critic_l2": l2.detach().cpu().numpy(),
    }
    return total_loss, stats

@torch.compile
def update_critic(
    batch: TensorDict, 
    critic: nn.Module, 
    critic_target: nn.Module, 
    actor: nn.Module, 
    gamma: float, 
    alpha: float | torch.Tensor,
    td_penalty: float,
    grad_target: float,
    grad_penalty: float,
    l2_penalty: float,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    optimizer.zero_grad()
    loss, stats = compute_critic_loss(
        batch, 
        critic, 
        critic_target, 
        actor, 
        gamma, 
        alpha, 
        td_penalty, 
        grad_target, 
        grad_penalty, 
        l2_penalty,
    )
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optimizer.step()
    return stats

def add_random_noise(x: torch.Tensor, noise: float, ratio: float, clip: bool = False):
    _x = x.clone()

    min_x = _x.min()
    max_x = _x.max()
    mask_len = int(ratio * len(x))
    with torch.no_grad():
        mask = torch.randperm(len(_x))[:mask_len]
        rand_x = torch.randn_like(_x, device=_x.device) * noise
        _x[mask] += rand_x[mask]
        if clip:
            _x[mask] = _x[mask].clip(min_x, max_x)
    return _x

def compute_actor_loss_awr(
    batch: TensorDict,
    critic: DoubleQNetwork,
    actor: nn.Module,
    alpha: torch.Tensor,
    action_mask_noise: float,
):
    obs = batch["observation"].clone()
    act_data = batch["action"].clone()
    
    # sample mask actions to prevent overfitting
    if action_mask_noise > 0:
        act_data = add_random_noise(act_data, action_mask_noise, ratio=0.5, clip=True)

    # compute advantage
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        q1_data, q2_data = critic(obs, act_data)
        q_data = torch.min(q1_data, q2_data)

        # compute baseline
        act, logp = sample_actor(obs, actor)
        q1, q2 = critic(obs, act)
        q = torch.min(q1, q2)
        v = q # - alpha * logp

        advantage = q_data - v
        weight = torch.exp(advantage / alpha).clip(max=100.)
    
    logp_data, _ = compute_action_likelihood(obs, act_data, actor, sample=False)

    actor_loss = -torch.mean(weight * logp_data) # advantage weighted regression
    
    stats = {
        "actor_loss": actor_loss.detach().cpu().numpy(),
        "alpha": alpha[0].detach().cpu().numpy(),
        "actor_value": q.data.mean().detach().cpu().numpy(),
        "advantage": advantage.data.mean().detach().cpu().numpy(),
        "advantage_weight": weight.data.mean().detach().cpu().numpy(),
    }
    return actor_loss, stats

def compute_actor_loss_sac(
    batch: TensorDict,
    critic: DoubleQNetwork,
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
    critic: DoubleQNetwork,
    actor: nn.Module,
    loss_type: str,
    alpha: torch.Tensor,
    log_alpha: torch.Tensor,
    alpha_target: torch.Tensor,
    tune_alpha: bool,
    action_mask_noise: float,
    actor_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    if loss_type == "sac":
        actor_optimizer.zero_grad()
        actor_loss, logp, actor_stats = compute_actor_loss_sac(
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
    else:
        actor_optimizer.zero_grad()
        actor_loss, actor_stats = compute_actor_loss_awr(
            batch, critic, actor, alpha, action_mask_noise
        )
        actor_loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
        actor_optimizer.step()
    
    return actor_stats


class Trainer(BaseTrainer):
    """Inverse soft Q learning

    Using Chi-square divergence and expert data only. The resulting update rule corresponds to regularized behavior cloning.
    
    We use advantage-weighted regression [2] to update the policy by default. 
    We also randomly mask out dataset actions and add small action noise to reduce policy overfitting.
    
    [1] IQ-Learn: Inverse soft-Q Learning for Imitation, Garg et al, 2021 (https://arxiv.org/abs/2106.12142)
    [2] Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning, Peng et al, 2019 (https://arxiv.org/abs/1910.00177)
    """
    def __init__(
        self, 
        config: IQLearnConfig,
        actor: TanhNormalActor,
        critic: DoubleQNetwork,
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

        self.actor = actor
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = nn.Parameter(
            np.log(config.alpha) * torch.ones(1, device=device), 
            requires_grad=config.tune_alpha,
        )
        self.alpha = self.log_alpha.data.exp()
        self.alpha_target = -torch.prod(torch.tensor(self.eval_env.action_spec.shape, device=device))

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
        }

        # store modules for checkpointing
        self._modules["actor"] = self.actor
        self._modules["critic"] = self.critic
        self._modules["critic_target"] = self.critic_target
        self._modules["log_alpha"] = self.log_alpha
        self._modules["obs_mean"] = nn.Parameter(obs_mean, requires_grad=False)
        self._modules["obs_std"] = nn.Parameter(obs_std, requires_grad=False)
    
    def take_policy_gradient_step(self, batch: TensorDict, global_step: int):
        actor_loss_type = self.config.actor_loss
        gamma = self.config.gamma
        alpha = self.alpha
        action_mask_noise = self.config.action_mask_noise
        td_penalty = self.config.td_penalty
        grad_target = self.config.grad_target
        grad_penalty = self.config.grad_penalty
        l2_penalty = self.config.l2_penalty
        tune_alpha = self.config.tune_alpha
        grad_clip = self.config.grad_clip

        critic_stats = update_critic(
            batch,
            self.critic,
            self.critic_target,
            self.actor,
            gamma,
            alpha,
            td_penalty,
            grad_target,
            grad_penalty,
            l2_penalty,
            self.optimizers["critic"],
            grad_clip,
        )

        actor_stats = update_actor(
            batch,
            self.critic,
            self.actor,
            actor_loss_type,
            alpha,
            self.log_alpha,
            self.alpha_target,
            tune_alpha,
            action_mask_noise,
            self.optimizers["actor"],
            self.optimizers["alpha"],
            grad_clip,
        )

        update_critic_target(self.critic, self.critic_target, self.config.polyak)
        self.alpha = self.log_alpha.detach().exp()
        
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

        policy_stats_epoch = []
        for _ in range(train_steps):
            batch = self.expert_buffer.sample(batch_size)
            policy_stats = self.take_policy_gradient_step(batch, global_step)
            policy_stats_epoch.append(policy_stats)

        policy_stats_epoch = tree_map(lambda *x: np.stack(x).mean(0), *policy_stats_epoch)
        return policy_stats_epoch
    
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
                
                    self.eval_action_likelihood(t)
                
                desc_str = f"epoch: {epoch}/{epochs}, step: {t}, eval_eps_return: {eval_return:.2f}"
                pbar.set_description(desc_str)

                # checkpoint handling
                if save_steps not in [None, -1] and (epoch + 1) % save_steps == 0:
                    self.save()
        
        pbar.close()
        self.close_envs()
        self.save()