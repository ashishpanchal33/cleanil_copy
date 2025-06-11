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
from cleanil.rl.critic import DoubleQNetwork, update_critic_target
from cleanil.rl.actor import TanhNormalActor, sample_actor, compute_action_likelihood
from cleanil.il.reward import compute_grad_penalty
from cleanil.utils import freeze_model_parameters

from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import Logger


@dataclass
class RECOILConfig(BaseTrainerConfig):
    # data args
    dataset: str = "halfcheetah-expert-v2"
    transition_dataset: str = "halfcheetah-medium-replay-v2"
    num_expert_trajs: int = 20
    transition_data_size: int = 100000
    train_ratio: float = 0.9
    upsample: bool = False

    # reward args
    expert_ratio: float = 0.5
    action_mask_noise: float = 0.1
    td_penalty: float = 0.8
    q_expert_target: float = 200.
    q_target_penalty: float = 1.
    grad_target: float = 100.
    grad_penalty: float = 0.

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
    expert_batch: TensorDict, 
    transition_batch: TensorDict,
    critic: nn.Module, 
    critic_target: nn.Module, 
    actor: nn.Module, 
    gamma: float, 
    alpha: float | torch.Tensor,
    td_penalty: float,
    q_expert_target: float,
    q_target_penalty: float,
    grad_target: float,
    grad_penalty: float,
) -> torch.Tensor:
    obs_expert = expert_batch["observation"]
    act_expert = expert_batch["action"]
    done_expert = expert_batch["terminated"].float()
    next_obs_expert = expert_batch["next"]["observation"]
    next_done_expert = expert_batch["next"]["terminated"].float()

    obs_transition = transition_batch["observation"]
    act_transition = transition_batch["action"]
    done_transition = transition_batch["terminated"].float()
    next_obs_transition = transition_batch["next"]["observation"]
    next_done_transition = transition_batch["next"]["terminated"].float()

    obs = torch.cat([obs_expert, obs_transition], dim=0)
    act = torch.cat([act_expert, act_transition], dim=0)
    done = torch.cat([done_expert, done_transition], dim=0)
    next_obs = torch.cat([next_obs_expert, next_obs_transition], dim=0)
    next_done = torch.cat([next_done_expert, next_done_transition], dim=0)

    critic_input = torch.cat([obs * (1 - done), done], dim=-1)
    critic_input_next = torch.cat([next_obs * (1 - next_done), next_done], dim=-1)
    critic_input_expert = torch.cat([obs_expert * (1 - done_expert), done_expert], dim=-1)
    critic_input_transition = torch.cat([obs_transition * (1 - done_transition), done_transition], dim=-1)

    with torch.no_grad():
        # sample fake action transition
        fake_act_transition, _ = sample_actor(obs_transition, actor)

        # sample fake action expert
        fake_act_expert, _ = sample_actor(obs_expert, actor)

        # compute value target
        next_act, logp = sample_actor(next_obs, actor)
        q1_next, q2_next = critic_target(critic_input_next, next_act)
        q_next = torch.min(q1_next, q2_next)
        v_next = q_next - alpha * logp
        q_target = (1 - done) * gamma * v_next

    # compute contrastive divergence loss
    q1_expert, q2_expert = critic(critic_input_expert, act_expert)
    q1_fake, q2_fake = critic(critic_input_expert, fake_act_expert)
    q1_transition, q2_transition = critic(critic_input_transition, fake_act_transition)
    cd1_loss = -(q1_expert.mean() - q1_fake.mean())
    cd2_loss = -(q2_expert.mean() - q2_fake.mean())
    cd_loss = (cd1_loss + cd2_loss) / 2

    # compute transition contrastive loss
    cd1_transition_loss = -(q1_expert.mean() - q1_transition.mean())
    cd2_transition_loss = -(q2_expert.mean() - q2_transition.mean())
    cd_transition_loss = (cd1_transition_loss + cd2_transition_loss) / 2

    # compute td loss
    q1, q2 = critic(critic_input, act)
    td1_loss = torch.pow(q1 - q_target, 2).mean()
    td2_loss = torch.pow(q2 - q_target, 2).mean()
    td_loss = (td1_loss + td2_loss) / 2

    # q expert target loss
    q_target_loss_1 = torch.pow(q1_expert - q_expert_target, 2).mean()
    q_target_loss_2 = torch.pow(q2_expert - q_expert_target, 2).mean()
    q_target_loss = (q_target_loss_1 + q_target_loss_2) / 2

    # q min hinge loss
    q_min = -q_expert_target * torch.ones_like(q1_transition, device=q1_transition.device)
    q_min_hinge_loss_1 = -torch.min(q1_transition, q_min).mean()
    q_min_hinge_loss_2 = -torch.min(q2_transition, q_min).mean()
    q_min_hinge_loss = (q_min_hinge_loss_1 + q_min_hinge_loss_2) / 2

    # compute grad penalty
    _done_expert = torch.zeros_like(done_expert, device=done_expert.device)
    _done_transition = torch.zeros_like(done_transition, device=done_transition.device)
    real_inputs = torch.cat([obs_expert, _done_expert, act_expert], dim=-1)
    fake_inputs = torch.cat([obs_transition, _done_transition, fake_act_transition], dim=-1)
    gp1, g_norm1 = compute_grad_penalty(real_inputs, fake_inputs, lambda x: critic.q1(x), grad_target)
    gp2, g_norm2 = compute_grad_penalty(real_inputs, fake_inputs, lambda x: critic.q2(x), grad_target)
    gp = (gp1 + gp2) / 2
    g_norm = (g_norm1 + g_norm2) / 2

    total_loss = (1 - td_penalty) * cd_transition_loss + \
        td_penalty * td_loss + \
        q_target_penalty * q_target_loss + \
        q_target_penalty * q_min_hinge_loss + \
        grad_penalty * gp
    
    stats = {
        "critic_loss": total_loss.detach().cpu().numpy(),
        "critic_cd_loss": cd_loss.detach().cpu().numpy(),
        "critic_cd_transition_loss": cd_transition_loss.detach().cpu().numpy(),
        "critic_td_loss": td_loss.detach().cpu().numpy(),
        "critic_expert_value": torch.min(q1_expert, q2_expert).mean().detach().cpu().numpy(),
        "critic_transition_value": torch.min(q1_transition, q2_transition).mean().detach().cpu().numpy(),
        "critic_fake_value": torch.min(q1_fake, q2_fake).mean().detach().cpu().numpy(),
        "critic_grad_pen": gp.detach().cpu().numpy(),
        "critic_grad_norm": g_norm.detach().cpu().numpy().mean(),
    }
    return total_loss, stats

@torch.compile
def update_critic(
    expert_batch: TensorDict, 
    transition_batch: TensorDict, 
    critic: nn.Module, 
    critic_target: nn.Module, 
    actor: nn.Module, 
    gamma: float, 
    alpha: float | torch.Tensor,
    td_penalty: float,
    q_expert_target: float,
    q_target_penalty: float,
    grad_target: float,
    grad_penalty: float,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    optimizer.zero_grad()
    loss, stats = compute_critic_loss(
        expert_batch, 
        transition_batch, 
        critic, 
        critic_target, 
        actor, 
        gamma, 
        alpha, 
        td_penalty,
        q_expert_target,
        q_target_penalty,
        grad_target,
        grad_penalty,
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

def compute_actor_loss(
    expert_batch: TensorDict,
    transition_batch: TensorDict,
    critic: DoubleQNetwork,
    actor: nn.Module,
    alpha: torch.Tensor,
    action_mask_noise: float,
):  
    act_transition = transition_batch["action"]

    # sample mask actions to prevent overfitting
    if action_mask_noise > 0:
        act_transition = add_random_noise(act_transition, action_mask_noise, ratio=0.5, clip=True)

    obs = torch.cat([expert_batch["observation"], transition_batch["observation"]], dim=0)
    done = torch.zeros(len(obs), 1, device=obs.device)
    act_data = torch.cat([expert_batch["action"], act_transition], dim=0)

    critic_input = torch.cat([obs, done], dim=-1)

    # compute advantage
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        q1_data, q2_data = critic(critic_input, act_data)
        q_data = torch.min(q1_data, q2_data)

        # compute baseline
        act, logp = sample_actor(obs, actor)
        q1, q2 = critic(critic_input, act)
        q = torch.min(q1, q2)
        v = q # - alpha * logp

        advantage = q_data - v
        weight = torch.exp(advantage / alpha).clip(max=100.)
    
    logp_data, _ = compute_action_likelihood(obs, act_data, actor, sample=False)

    actor_loss = -torch.mean(weight * logp_data) # advantage weighted regression
    
    expert_weight, transition_weight = torch.chunk(weight, 2, dim=0)
    expert_advantage, transition_advantage = torch.chunk(advantage, 2, dim=0)
    stats = {
        "actor_loss": actor_loss.detach().cpu().numpy(),
        "alpha": alpha[0].detach().cpu().numpy(),
        "actor_value": q.data.mean().detach().cpu().numpy(),
        "advantage": advantage.data.mean().detach().cpu().numpy(),
        "advantage_weight": weight.data.mean().detach().cpu().numpy(),
        "advantage_expert": expert_advantage.data.mean().detach().cpu().numpy(),
        "advantage_transition": transition_advantage.data.mean().detach().cpu().numpy(),
        "advantage_weight_expert": expert_weight.data.mean().detach().cpu().numpy(),
        "advantage_weight_transition": transition_weight.data.mean().detach().cpu().numpy(),
    }
    return actor_loss, stats

@torch.compile
def update_actor(
    expert_batch: TensorDict,
    transition_batch: TensorDict,
    critic: DoubleQNetwork,
    actor: nn.Module,
    alpha: torch.Tensor,
    action_mask_noise: float,
    actor_optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    actor_optimizer.zero_grad()
    actor_loss, stats = compute_actor_loss(
        expert_batch, transition_batch, critic, actor, alpha, action_mask_noise
    )
    actor_loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    actor_optimizer.step()
    return stats

def set_last_layer_bias(model: nn.Module, target: float):
    model[-1].bias.data = target * torch.ones_like(model[-1].bias.data, device=model[-1].bias.device)


class Trainer(BaseTrainer):
    """Relaxed coverage for off-policy imitation learning

    Dual-Q form of RECOIL with Chi-square divergence. 
    The resulting update rule corresponds roughly to regularized behavior cloning on expert-transition data mixture. 
    
    We use a few tricks from the official implementation: 
    1) We add a squared error loss to regress expert value to a target value, similar to LS-IQ [2]. We use a hinge loss to lower bound suboptimal data value.
    2) We use advantage-weighted regression [3] to update the policy. We also randomly mask out dataset actions to reduce policy overfitting.

    [1] Dual RL: Unification and New Methods for Reinforcement and Imitation Learning, Sikchi et al, 2023 (https://arxiv.org/abs/2302.08560)
    [2] LS-IQ: Implicit Reward Regularization for Inverse Reinforcement Learning, Al-Hafez et al, 2023 (https://arxiv.org/abs/2303.00599)
    [3] Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning, Peng et al, 2019 (https://arxiv.org/abs/1910.00177)
    """
    def __init__(
        self, 
        config: RECOILConfig,
        actor: TanhNormalActor,
        critic: DoubleQNetwork,
        expert_buffer: ReplayBuffer,
        transition_buffer: ReplayBuffer,
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
        self.transition_buffer = transition_buffer
        self.eval_buffer = eval_buffer

        self.actor = actor
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = nn.Parameter(
            np.log(config.alpha) * torch.ones(1, device=device), 
            requires_grad=False,
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
    
    def take_policy_gradient_step(self, expert_batch: TensorDict, transition_batch: TensorDict, global_step: int):
        gamma = self.config.gamma
        alpha = self.alpha
        action_mask_noise = self.config.action_mask_noise
        td_penalty = self.config.td_penalty
        q_expert_target = self.config.q_expert_target
        q_target_penalty = self.config.q_target_penalty
        grad_target = self.config.grad_target
        grad_penalty = self.config.grad_penalty
        grad_clip = self.config.grad_clip

        critic_stats = update_critic(
            expert_batch,
            transition_batch,
            self.critic,
            self.critic_target,
            self.actor,
            gamma,
            alpha,
            td_penalty,
            q_expert_target,
            q_target_penalty,
            grad_target,
            grad_penalty,
            self.optimizers["critic"],
            grad_clip,
        )

        actor_stats = update_actor(
            expert_batch,
            transition_batch,
            self.critic,
            self.actor,
            alpha,
            action_mask_noise,
            self.optimizers["actor"],
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
        expert_batch_size = int(self.config.expert_ratio * batch_size)
        transition_batch_size = batch_size - expert_batch_size

        policy_stats_epoch = []
        for _ in range(train_steps):
            expert_batch = self.expert_buffer.sample(expert_batch_size)
            transition_batch = self.transition_buffer.sample(transition_batch_size)
            policy_stats = self.take_policy_gradient_step(expert_batch, transition_batch, global_step)
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