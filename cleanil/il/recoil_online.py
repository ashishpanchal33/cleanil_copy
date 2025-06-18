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
from cleanil.utils import freeze_model_parameters, concat_tensordict_on_shared_keys

from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import Logger


@dataclass
class RECOILOnlineConfig(BaseTrainerConfig):
    # data args
    dataset: str = "halfcheetah-expert-v2"
    num_expert_trajs: int = 10
    train_ratio: float = 0.9

    # reward args
    q_max: float = 1000.
    use_double: bool = True
    use_done: bool = False
    ibc_penalty: float = 0.1
    td_penalty: float = 0.2
    grad_target: float = 200.
    grad_penalty: float = 10.
    decouple_loss: bool = False

    # rl args
    hidden_dims: list = (256, 256)
    activation: str = "silu"
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.1
    tune_alpha: float = False
    batch_size: int = 256
    real_ratio: float = 0.5
    policy_train_steps: int = 1
    policy_update_every: int = 1
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    grad_clip: float = 1000.

    # train args
    buffer_size: int = 1e6
    max_eps_steps: int = 1000
    epochs: int = 1000
    steps_per_epoch: int = 1000
    update_after: int = 2000
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
    done_expert = expert_batch["next"]["terminated"].float()

    obs_transition = transition_batch["observation"]
    done_transition = transition_batch["next"]["terminated"].float()

    obs = td_batch["observation"]
    act = td_batch["action"]
    done = td_batch["terminated"].float()
    next_obs = td_batch["next"]["observation"]
    next_done = td_batch["next"]["terminated"].float()

    with torch.no_grad():
        # sample fake action expert
        fake_act_expert, _ = sample_actor(obs_expert, actor)

        # sample fake action transition
        fake_act_transition, _ = sample_actor(obs_transition, actor)

        # compute value target
        next_act, logp = sample_actor(next_obs, actor)
        q1_next, q2_next = critic_target.forward(next_obs, next_act, next_done, clip=True)
        q_next = torch.min(q1_next, q2_next)
        v_next = q_next - alpha * logp
        q_target = gamma * v_next

    q1_expert, q2_expert = critic.forward(obs_expert, act_expert, done_expert, clip=True)
    q1_fake, q2_fake = critic.forward(obs_expert, fake_act_expert, done_expert, clip=True)
    q1_transition, q2_transition = critic.forward(obs_transition, fake_act_transition, done_transition, clip=True)

    # compute ibc loss
    ibc_loss_1 = -(q1_expert.mean() - q1_fake.mean())
    ibc_loss_2 = -(q2_expert.mean() - q2_fake.mean())
    ibc_loss = (ibc_loss_1 + ibc_loss_2) / 2

    # compute transition contrastive loss
    cd_loss_1 = -(q1_expert.mean() - q1_transition.mean())
    cd_loss_2 = -(q2_expert.mean() - q2_transition.mean())
    cd_loss = (cd_loss_1 + cd_loss_2) / 2

    # compute td loss
    q1, q2 = critic.forward(obs, act, done, clip=True)
    td_loss_1 = torch.pow(q1 - q_target, 2).mean()
    td_loss_2 = torch.pow(q2 - q_target, 2).mean()
    td_loss = (td_loss_1 + td_loss_2) / 2

    _cd_loss = ibc_penalty * ibc_loss + (1 - ibc_penalty) * cd_loss
    total_loss = (1 - td_penalty) * _cd_loss + td_penalty * td_loss

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
        _act = torch.cat([act_expert, fake_act_transition], dim=0)
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
    expert_batch: TensorDict,
    transition_batch: TensorDict,
    critic: CriticWrapper,
    actor: nn.Module,
    alpha: torch.Tensor,
):  
    obs = torch.cat([expert_batch["observation"], transition_batch["observation"]], dim=0)
    done = torch.zeros(len(obs), 1, device=obs.device) # dummy done

    act, logp = sample_actor(obs, actor)
    q1, q2 = critic.forward(obs, act, done, clip=True)
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
    expert_batch: TensorDict,
    transition_batch: TensorDict,
    critic: DoubleQNetwork,
    actor: nn.Module,
    alpha: torch.Tensor,
    log_alpha: torch.Tensor,
    alpha_target: torch.Tensor,
    tune_alpha: bool,
    q_max: float,
    actor_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
):
    actor_optimizer.zero_grad()
    actor_loss, logp, actor_stats = compute_actor_loss(
        expert_batch, transition_batch, critic, actor, alpha, q_max
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


class Trainer(BaseTrainer):
    """Online relaxed coverage for off-policy imitation learning
    
    This is a modification of the RECOIL algorithm for online imitation learning. 
    We essentially replace the offline dataset with the online replay buffer.
    The resulting algorithm is similar to online IQ-Learn [2].

    [1] Dual RL: Unification and New Methods for Reinforcement and Imitation Learning, Sikchi et al, 2023 (https://arxiv.org/abs/2302.08560)
    [2] IQ-Learn: Inverse soft-Q Learning for Imitation, Garg et al, 2021 (https://arxiv.org/abs/2106.12142)
    """
    def __init__(
        self, 
        config: RECOILOnlineConfig,
        actor: TanhNormalActor,
        critic: DoubleQNetwork,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        expert_buffer: ReplayBuffer,
        eval_buffer: ReplayBuffer,
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
        self.eval_buffer = eval_buffer

        critic = CriticWrapper(critic, config.use_double, config.use_done, config.q_max)

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
        }

        # store modules for checkpointing
        self._modules["actor"] = self.actor
        self._modules["critic"] = self.critic
        self._modules["critic_target"] = self.critic_target
        self._modules["log_alpha"] = self.log_alpha
        self._modules["obs_mean"] = nn.Parameter(obs_mean, requires_grad=False)
        self._modules["obs_std"] = nn.Parameter(obs_std, requires_grad=False)
    
    def take_policy_gradient_step(self, expert_batch: TensorDict, transition_batch: TensorDict, td_batch: TensorDict, global_step: int):
        gamma = self.config.gamma
        alpha = self.alpha
        q_max = self.config.q_max
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
            expert_batch,
            transition_batch,
            self.critic,
            self.actor,
            alpha,
            self.log_alpha,
            self.alpha_target,
            tune_alpha,
            q_max,
            self.optimizers["actor"],
            self.optimizers["alpha"],
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
        update_after = config.update_after
        update_every = config.policy_update_every
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

            # train policy
            if (t + 1) > update_after and (t - update_after + 1) % update_every == 0:
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