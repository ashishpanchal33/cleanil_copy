


import time
#import copy
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import numpy as np
#import torch
#import torch.nn as nn
from optree import tree_map
from cleanil.base_trainer import BaseTrainer, BaseTrainerConfig

from cleanil.rl.critic_discrete import DoubleQNetwork, compute_q_target, update_critic_target
#from cleanil.rl.actor import TanhNormalActor, sample_actor, compute_action_likelihood






import torch
import torch.nn as nn
from cleanil.il.reward import compute_grad_penalty


from cleanil.utils import freeze_model_parameters, compute_parameter_l2

from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import Logger




import copy
from dataclasses import dataclass
import torch
import torch.nn as nn
from cleanil.il.iqlearn import (
    IQLearnConfig, #compute_critic_loss, update_critic, 
    add_random_noise#, OfflineTrainer
)
#from cleanil.rl.actor_discrete import (
#    CategoricalActor, sample_discrete_actor, compute_discrete_action_likelihood
#)
#from cleanil.rl.critic import DoubleQNetwork




@dataclass
class IQLearnConfig(BaseTrainerConfig):
    # data args
    dataset: str = "halfcheetah-expert-v2"
    num_expert_trajs: int = 20
    train_ratio: float = 0.9

        
    
    obs_dim : int = 4,
    act_dim : int = 1,
    num_actions: int = 2,

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




import torch
import torch.nn as nn
from cleanil.il.reward import compute_grad_penalty
from cleanil.utils import compute_parameter_l2

def compute_critic_loss_discrete(
    batch: TensorDict,
    critic: nn.Module,
    critic_target: nn.Module,
    num_actions: int,
    gamma: float,
    alpha: torch.Tensor,
    td_penalty: float,
    grad_target: float,
    grad_penalty: float,
    l2_penalty: float,
) -> (torch.Tensor, dict):
    obs      = batch["observation"]                      # [B, obs_dim]
    act      = batch["action"].squeeze(-1).long()        # [B]
    done     = batch["terminated"].float().unsqueeze(-1) # [B,1]
    next_obs = batch["next"]["observation"]              # [B, obs_dim]

    # 1. Target Q for next states (all actions)
    q1_next_all, q2_next_all = critic_target(next_obs)   # [B, num_actions] each
    q_next_all = torch.min(q1_next_all, q2_next_all)     # [B, num_actions]
    v_next = alpha * torch.logsumexp(q_next_all / alpha, dim=-1, keepdim=True)  # [B,1]

    # 2. TD target
    q_target = (1 - done) * gamma * v_next               # [B,1]

    # 3. Q(s,a) for real actions
    q1_real, q2_real = critic(obs, act.unsqueeze(-1))    # [B,1] each
    q_real = torch.min(q1_real, q2_real)                 # [B,1]

    # 4. Sample fake actions
    with torch.no_grad():
        q1_all, q2_all = critic(obs)                     # [B, num_actions] each
        q_all = torch.min(q1_all, q2_all)                # [B, num_actions]
        fake_probs = torch.softmax(q_all / alpha, dim=-1)# [B, num_actions]
        fake_act = torch.multinomial(fake_probs, 1).squeeze(-1)  # [B]
    q1_fake, q2_fake = critic(obs, fake_act.unsqueeze(-1))     # [B,1]
    q_fake = torch.min(q1_fake, q2_fake)                       # [B,1]

    # 5. Contrastive divergence
    cd_loss = - (q_real.mean() - q_fake.mean())

    # 6. TD loss
    td_loss = (q_real - q_target).pow(2).mean()

    # 7. Gradient penalty
    if 0:
        real_inputs = torch.cat([obs, act.unsqueeze(-1).float()], dim=-1)
        fake_inputs = torch.cat([obs, fake_act.unsqueeze(-1).float()], dim=-1)
        gp1, g_norm1 = compute_grad_penalty(real_inputs, fake_inputs,
                                            lambda x: critic.q1(x), grad_target)
        gp2, g_norm2 = compute_grad_penalty(real_inputs, fake_inputs,
                                            lambda x: critic.q2(x), grad_target)
        gp     = (gp1 + gp2) / 2
        g_norm = (g_norm1 + g_norm2) / 2

    else:
        gp,g_norm = 0.0,0.0


    

    # 8. L2 penalty
    l2 = compute_parameter_l2(critic)

    # 9. Total loss
    total_loss = ((1 - td_penalty) * cd_loss
                  + td_penalty * td_loss
                  + grad_penalty * gp
                  + l2_penalty  * l2)

    stats = {
        "critic_loss":      total_loss.item(),
        "critic_cd_loss":   cd_loss.item(),
        "critic_td_loss":   td_loss.item(),
        "critic_grad_pen":  gp,#gp.item(),
        "critic_grad_norm": g_norm,#g_norm.item(),
        "critic_l2":        l2.item(),
    }
    return total_loss, stats





@torch.compile
def update_critic_discrete(
    batch: TensorDict, 
    critic: nn.Module, 
    critic_target: nn.Module, 
    num_actions: int,
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
    loss, stats = compute_critic_loss_discrete(
        batch, 
        critic, 
        critic_target, 
        num_actions,
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




class DiscreteOfflineTrainer(BaseTrainer):
    def __init__(self, config, critic, expert_buffer, eval_buffer, 
                 obs_mean, obs_std, num_actions, logger=None, device="cpu"):
        BaseTrainer.__init__(self, config, None, None, None, logger, device)
        
        self.config = config
        self.expert_buffer = expert_buffer
        self.eval_buffer = eval_buffer
        self.num_actions = num_actions
        
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.alpha = torch.tensor(config.alpha, device=device)
        
        freeze_model_parameters(self.critic_target)
        
        self.optimizers = {
            "critic": torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic),
        }
        
        # Store modules for checkpointing
        self._modules["critic"] = self.critic
        self._modules["critic_target"] = self.critic_target
        self._modules["obs_mean"] = nn.Parameter(obs_mean, requires_grad=False)
        self._modules["obs_std"] = nn.Parameter(obs_std, requires_grad=False)
    
    def take_policy_gradient_step(self, batch, global_step: int):
        critic_stats = update_critic_discrete(
            batch, self.critic, self.critic_target, self.num_actions,
            self.config.gamma, self.alpha, self.config.td_penalty,
            self.config.grad_target, self.config.grad_penalty, 
            self.config.l2_penalty, self.optimizers["critic"], 
            self.config.grad_clip,
        )
        
        # Update target network
        from cleanil.rl.critic_discrete import update_critic_target
        update_critic_target(self.critic, self.critic_target, self.config.polyak)
        
        if self.logger is not None:
            for metric_name, metric_value in critic_stats.items():
                self.logger.log_scalar(f"critic/{metric_name}", metric_value, global_step)
        
        return critic_stats
    
    # Use the discrete evaluation functions above
    #eval_action_likelihood = eval_discrete_action_likelihood
    #eval_recovered_rewards = eval_recovered_rewards_discrete

    
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
    


    def train(self):
        # Same as OfflineTrainer but no actor updates
        config = self.config
        epochs = config.epochs
        steps_per_epoch = config.steps_per_epoch
        save_steps = config.save_steps

        total_steps = epochs * steps_per_epoch
        start_time = time.time()
        
        epoch = 0
        pbar = tqdm(range(total_steps))
        for t in pbar:
            policy_stats_epoch = self.train_policy_epoch(t)

            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch
                
                self.eval_discrete_action_likelihood(t)
                self.eval_recovered_rewards_discrete(t)
                
                speed = (t + 1) / (time.time() - start_time)
                if self.logger is not None:
                    self.logger.log_scalar("train_speed", speed, t)
                
                desc_str = f"epoch: {epoch}/{epochs}, step: {t}"
                pbar.set_description(desc_str)

                if save_steps not in [None, -1] and (epoch + 1) % save_steps == 0:
                    self.save()
        
        pbar.close()
        self.save()



    def eval_discrete_action_likelihood(self, global_step: int, num_samples=1000):
        def eval_func(batch):
            obs = batch["observation"]
            act = batch["action"].squeeze(-1).long()  # [batch]
                
            # 1. Get Q-values for all actions from both networks
            q1_all, q2_all = self.critic(obs)                   # each [B, num_actions]
            q_all = torch.min(q1_all, q2_all)                   # [B, num_actions]
                
            # Convert to policy probabilities
            logp_all = torch.log_softmax(q_all, dim=-1)  # [batch, num_actions]
            logp = logp_all.gather(1, act.unsqueeze(-1))  # [batch, 1]
            
            # Accuracy instead of MAE
            pred_act = logp_all.argmax(dim=-1)  # [batch]
            accuracy = (pred_act == act).float().mean()
            
            return logp, accuracy

        
        with torch.no_grad():
            train_batch = self.expert_buffer.sample(num_samples)
            eval_batch = self.eval_buffer.sample(num_samples)
            train_logp, train_acc = eval_func(train_batch)
            eval_logp, eval_acc = eval_func(eval_batch)
    
        stats = {
            "train_act_logp": train_logp.mean().detach().cpu().numpy(), 
            "train_act_accuracy": train_acc.detach().cpu().numpy(),
            "eval_act_logp": eval_logp.mean().detach().cpu().numpy(), 
            "eval_act_accuracy": eval_acc.detach().cpu().numpy(),
        }
        
        if self.logger is not None:
            for metric_name, metric_value in stats.items():
                self.logger.log_scalar(metric_name, metric_value, global_step)

    
    def eval_recovered_rewards_discrete(self, global_step: int, num_samples=1000):
        with torch.no_grad():
            train_batch = self.expert_buffer.sample(num_samples)
            eval_batch = self.eval_buffer.sample(num_samples)

            def recover_rewards(batch):
                obs      = batch["observation"]                       # [B, obs_dim]
                act      = batch["action"].squeeze(-1).long()         # [B]
                next_obs = batch["next"]["observation"]               # [B, obs_dim]
    
                # Q(s, a)
                q1_real, q2_real = self.critic(obs, act.unsqueeze(-1))  # [B,1], [B,1]
                q_real = torch.min(q1_real, q2_real)                    # [B,1]
    
                # V(s') = alpha * logsumexp(min Q(next_obs, a')/alpha)
                q1_next_all, q2_next_all = self.critic(next_obs)        # [B, num_actions]
                q_next_all = torch.min(q1_next_all, q2_next_all)       # [B, num_actions]
                v_next = self.alpha * torch.logsumexp(q_next_all / self.alpha, dim=-1, keepdim=True)  # [B,1]
    
                # Recovered reward per transition
                return (q_real - self.config.gamma * v_next).squeeze(-1)  # [B]
    
    
            
            
            train_rewards = recover_rewards(train_batch)
            eval_rewards = recover_rewards(eval_batch)
            
            stats = {
                "recovered_train_reward_mean": train_rewards.mean().cpu().numpy(),
                "recovered_train_reward_std": train_rewards.std().cpu().numpy(),
                "recovered_eval_reward_mean": eval_rewards.mean().cpu().numpy(),
                "recovered_eval_reward_std": eval_rewards.std().cpu().numpy(),
            }
            
            if self.logger is not None:
                for name, value in stats.items():
                    self.logger.log_scalar(name, value, global_step)

