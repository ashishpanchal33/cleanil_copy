import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributions as torch_dist
from optree import tree_map
from cleanil.utils import soft_clamp, get_activation
from cleanil.data import normalize, denormalize
from tensordict import TensorDict
from torchrl.record.loggers import Logger


@dataclass
class EnsembleConfig:
    ensemble_dim: int = 7
    topk: int = 5
    hidden_dims: list = (200, 200, 200)
    activation: str = "silu"
    min_std: float = 0.04
    max_std: float = 1.6
    decays: list = (0.000025, 0.00005, 0.000075, 0.000075, 0.0001)


class EnsembleLinear(nn.Module):
    """Ensemble version of nn.Linear"""
    def __init__(self, input_dim: int, output_dim: int, ensemble_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_dim = ensemble_dim

        self.weight = nn.Parameter(torch.zeros(ensemble_dim, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_dim, output_dim))
        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))
    
    def __repr__(self):
        s = "{}(input_dim={}, output_dim={}, ensemble_dim={})".format(
            self.__class__.__name__, self.input_dim, self.output_dim, self.ensemble_dim,
        )
        return s

    def forward(self, x: torch.Tensor):
        """Output size=[..., ensemble_dim, output_dim]"""
        out = torch.einsum("kio, ...ki -> ...ko", self.weight, x) + self.bias
        return out
    
    
class EnsembleMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        ensemble_dim: int, 
        hidden_dims: list[int], 
        activation: nn.Module,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_dim = ensemble_dim
        self.hidden_dims = hidden_dims

        last_dim = input_dim
        layers = []
        for h in hidden_dims:
            layers.append(EnsembleLinear(last_dim, h, ensemble_dim))
            layers.append(activation())
            last_dim = h
        layers.append(EnsembleLinear(last_dim, output_dim, ensemble_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for inputs that don't have an ensemble dimension
        Args:
            x (torch.tensor): input batch. size=[batch_size, input_dim]

        Outputs:
            x (torch.tensor): output batch. size=[batch_size, k, output_dim]
        """
        x = x.unsqueeze(-2).repeat_interleave(self.ensemble_dim, dim=-2)
        return self.layers(x)
    
    def forward_separete(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for inputs that do have an ensemble dimension
        
        Args:
            x (torch.tensor): size=[..., ensemble_dim, input_dim]

        Returns:
            out (torch.tensor): size=[..., ensemble_dim, output_dim] 
        """
        return self.layers(x)
    

class StandardScaler(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.variance = nn.Parameter(torch.ones(input_dim), requires_grad=False)
    
    def __repr__(self):
        s = "{}(input_dim={})".format(self.__class__.__name__, self.input_dim)
        return s
    
    def fit(self, x: torch.Tensor):
        """Update stats from torch tensor"""
        device = self.mean.device
        self.mean.data = x.mean(0).to(device)
        self.variance.data = x.var(0).to(device)
        
    def transform(self, x: torch.Tensor):
        """Normalize inputs torch tensor"""
        return normalize(x, self.mean, self.variance)
    
    def inverse_transform(self, x_norm: torch.Tensor):
        """Denormalize inputs torch tensor"""
        return denormalize(x_norm, self.mean, self.variance)


class EnsembleDynamics(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        out_dim: int,
        config: EnsembleConfig,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = out_dim
        self.ensemble_dim = config.ensemble_dim
        self.topk = config.topk
        self.config = config
        
        self.input_scaler = StandardScaler(self.obs_dim + self.act_dim)
        self.mlp = EnsembleMLP(
            obs_dim + act_dim, 
            self.out_dim * 2, 
            config.ensemble_dim, 
            config.hidden_dims,  
            get_activation(config.activation),
        )

        topk_dist = torch.ones(self.ensemble_dim) / self.ensemble_dim # model selection distribution
        self.topk_dist = nn.Parameter(topk_dist, requires_grad=False)
        self.min_lv = nn.Parameter(np.log(config.min_std**2) * torch.ones(self.out_dim), requires_grad=True)
        self.max_lv = nn.Parameter(np.log(config.max_std**2) * torch.ones(self.out_dim), requires_grad=True)
    
    @torch.compile
    def compute_stats(self, inputs: torch.Tensor):
        """Compute prediction stats 
        
        Args:
            inputs (torch.tensor): inputs. size=[..., obs_dim + act_dim]

        Returns:
            mu (torch.tensor): prediction mean. size=[..., ensemble_dim, out_dim]
            lv (torch.tensor): prediction log variance. size=[..., ensemble_dim, out_dim]
        """
        inputs_norm = self.input_scaler.transform(inputs)
        mu, lv_ = torch.chunk(self.mlp.forward(inputs_norm), 2, dim=-1)
        lv = soft_clamp(lv_, self.min_lv, self.max_lv)
        return mu, lv
    
    def compute_stats_separate(self, inputs: torch.Tensor):
        """Compute prediction stats for inputs with an ensemble dim 
        
        Args:
            inputs (torch.tensor): inputs. size=[..., ensemble_dim, obs_dim + act_dim]

        Returns:
            mu (torch.tensor): prediction mean. size=[..., ensemble_dim, out_dim]
            lv (torch.tensor): prediction log variance. size=[..., ensemble_dim, out_dim]
        """
        inputs_norm = self.input_scaler.transform(inputs)
        mu, lv_ = torch.chunk(self.mlp.forward_separete(inputs_norm), 2, dim=-1)
        lv = soft_clamp(lv_, self.min_lv, self.max_lv)
        return mu, lv
    
    def get_dist(self, obs: torch.Tensor, act: torch.Tensor):
        """Compute prediction distribution classes 
        
        Returns:
            dist (torch_dist.Normal): prediction distribution
        """
        obs_act = torch.cat([obs, act], dim=-1)
        mu, lv = self.compute_stats(obs_act)
        std = torch.exp(0.5 * lv)
        return torch_dist.Normal(mu, std)
    
    def compute_log_prob(self, obs: torch.Tensor, act: torch.Tensor, target: torch.Tensor):
        """Compute ensemble log probability 
        
        Args:
            obs (torch.tensor): observations. size=[..., obs_dim]
            act (torch.tensor): actions. size=[..., act_dim]
            target (torch.tensor): targets. size=[..., out_dim]

        Returns:
            logp_obs (torch.tensor): ensemble log probabilities of targets. size=[..., ensemble_dim, 1]
        """
        dist = self.get_dist(obs, act)
        logp = dist.log_prob(target.unsqueeze(-2)).sum(-1, keepdim=True)
        return logp
    
    def compute_mixture_log_prob(self, obs: torch.Tensor, act: torch.Tensor, target: torch.Tensor):
        """Compute log marginal probability 
        
        Args:
            obs (torch.tensor): observations. size=[..., obs_dim]
            act (torch.tensor): actions. size=[..., act_dim]
            next_obs (torch.tensor): targets. size=[..., out_dim]

        Returns:
            mixture_logp (torch.tensor): log marginal probabilities of targets. size=[..., 1]
        """
        log_elites = torch.log(self.topk_dist + 1e-6)
        log_elites[self.topk_dist == 0.] -= 1e6
        logp = self.compute_log_prob(obs, act, target)
        mixture_logp = torch.logsumexp(logp + log_elites.unsqueeze(-1), dim=-2)
        return mixture_logp
    
    def sample_dist(self, obs, act, sample_mean=False):
        """Ancestral sampling from ensemble
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., obs_dim]
            act (torch.tensor): normaized actions. size=[..., act_dim]
            sample_mean (bool, optional): whether to sample mean. Default=False

        Returns:
            out (torch.tensor): predictions sampled from ensemble member in topk_dist. size=[..., out_dim]
        """
        dist = self.get_dist(obs, act)
        if not sample_mean:
            out = dist.rsample()
        else:
            out = dist.mean
        
        # randomly select from top models
        ensemble_idx = torch_dist.Categorical(self.topk_dist).sample(obs.shape[:-1]).unsqueeze(-1)
        ensemble_idx_obs = ensemble_idx.unsqueeze(-1).repeat_interleave(self.out_dim, dim=-1) # duplicate alone feature dim

        out = torch.gather(out, -2, ensemble_idx_obs).squeeze(-2)
        return out
    
    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mu, lv = self.compute_stats_separate(inputs)
        inv_var = torch.exp(-lv)

        mse_loss = torch.mean(torch.pow(mu - targets, 2) * inv_var, dim=-1).mean(0)
        var_loss = torch.mean(lv, dim=-1).mean(0)

        clip_lv_loss = 0.001 * (self.max_lv.sum() - self.min_lv.sum())
        decay_loss = self.compute_decay_loss()

        loss = (
            mse_loss.mean() + var_loss.mean() \
            + clip_lv_loss \
            + decay_loss
        )
        return loss
    
    def compute_decay_loss(self):
        decays = self.config.decays
        i, loss = 0, 0
        for n, p in self.mlp.named_parameters():
            if "weight" in n:
                loss += decays[i] * torch.sum(p ** 2) / 2.
                i += 1
        return loss

    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Compute mae for each ensemble member
        
        Returns:
            stats (dict): MAE dict with fields [mae_0, ..., mae_{ensemble_dim}, mae]
        """
        with torch.no_grad():
            mu, lv = self.compute_stats(inputs)
            std = torch.exp(0.5 * lv)
            dist = torch_dist.Normal(mu, std)
            logp = dist.log_prob(targets.unsqueeze(-2)).mean(-1)
        
        mae = torch.abs(mu - targets.unsqueeze(-2)).mean((0, 2))
        loss = -torch.mean(logp, dim=0)
        
        maes = {f"mae_{i}": mae[i].cpu().data.item() for i in range(self.ensemble_dim)}
        losses = {f"nlogp_{i}": loss[i].cpu().data.item() for i in range(self.ensemble_dim)}
        stats = {**maes, **losses}
        stats["mae"] = mae.mean().cpu().data.item()
        stats["nlogp"] = loss.mean().cpu().data.item()
        return stats
    
    def update_topk_dist(self, stats: dict[float]):
        """Update top k model selection distribution"""
        maes = [stats[f"mae_{i}"] for i in range(self.ensemble_dim)]
        idx_topk = np.argsort(maes)[:self.topk]
        topk_dist = np.zeros(self.ensemble_dim)
        topk_dist[idx_topk] = 1./self.topk
        self.topk_dist.data = torch.from_numpy(topk_dist).to(torch.float32).to(self.topk_dist.device)


def parse_ensemble_params(ensemble: EnsembleDynamics) -> list[dict[torch.Tensor]]:
    """Parse ensemble parameters into list of size ensemble_dim 
    
    Returns:
        params_list (list): list of named parameters dictionaries
    """
    ensemble_dim = ensemble.ensemble_dim
    params_list = [{} for _ in range(ensemble_dim)]
    for n, p in ensemble.named_parameters():
        if "weight" in n or "bias" in n:
            for i in range(ensemble_dim):
                params_list[i].update({n: p[i].data.clone()})
    return params_list

def set_ensemble_params(ensemble: EnsembleDynamics, params_list: list[dict[torch.Tensor]]):
    """Set ensemble parameters from list of size ensemble_dim 
    
    Args:
        ensemble (EnsembleDyanmics): EnsembleDynamics object
        params_list (list): list of named parameters dictionaries
    """
    ensemble_dim = ensemble.ensemble_dim
    for n, p in ensemble.named_parameters():
        if "weight" in n or "bias" in n:
            p.data = torch.stack([params_list[i][n] for i in range(ensemble_dim)])

def remove_non_topk_members(ensemble: EnsembleDynamics, config: EnsembleConfig) -> EnsembleDynamics:
    topk_dist = ensemble.topk_dist
    idx_to_keep = torch.where(topk_dist > 0.)[0]

    def reduce_state_dict(state_dict: dict[torch.Tensor]):
        for k, v in state_dict.items():
            if "mlp." in k or "topk_dist" in k:
                state_dict[k] = v[idx_to_keep]
    
    state_dict = ensemble.state_dict()
    reduce_state_dict(state_dict)

    config.ensemble_dim = len(idx_to_keep)
    reduced_ensemble = EnsembleDynamics(
        ensemble.obs_dim,
        ensemble.act_dim,
        ensemble.out_dim,
        config,
    )
    reduced_ensemble.load_state_dict(state_dict)
    return reduced_ensemble

def remove_reward_head(state_dict: dict[torch.Tensor], obs_dim: int):
    max_layer_number = max([
        int(k.replace("mlp.layers.", "").split(".")[0]) for k in state_dict.keys() if "mlp.layers." in k
    ])
    weight_key = f"mlp.layers.{max_layer_number}.weight"
    bias_key = f"mlp.layers.{max_layer_number}.bias"
    head_weight = state_dict[weight_key]
    head_bias = state_dict[bias_key]
    
    if head_weight.shape[-1] == ((obs_dim + 1) * 2):
        head_weight_mu, head_weight_lv = torch.chunk(head_weight, 2, dim=-1)
        head_bias_mu, head_bias_lv = torch.chunk(head_bias, 2, dim=-1)

        head_weight_mu = head_weight_mu[..., :-1]
        head_weight_lv = head_weight_lv[..., :-1]
        head_bias_mu = head_bias_mu[..., :-1]
        head_bias_lv = head_bias_lv[..., :-1]

        head_weight = torch.cat([head_weight_mu, head_weight_lv], dim=-1)
        head_bias = torch.cat([head_bias_mu, head_bias_lv], dim=-1)

        state_dict[weight_key] = head_weight
        state_dict[bias_key] = head_bias
        state_dict["min_lv"] = state_dict["min_lv"][:-1]
        state_dict["max_lv"] = state_dict["max_lv"][:-1]
    else:
        raise ValueError("last layer dimensinon should be (obs_dim + 1) * 2")
    return state_dict
        
def format_samples_for_training(
    data: TensorDict, 
    dynamics: EnsembleDynamics,
    pred_rwd: bool, 
    eval_ratio: float, 
    max_eval_num: float,
):
    """Formate transition samples into inputs and targets and do train test split"""
    obs = data["observation"]
    act = data["action"]
    rwd = data["next"]["reward"]
    next_obs = data["next"]["observation"]

    inputs = torch.cat([obs, act], dim=-1)
    targets = next_obs - obs
    if pred_rwd:
        targets = torch.cat([targets, rwd], dim=-1)

    # train test split
    num_eval = min(int(len(inputs) * eval_ratio), max_eval_num)
    permutation = np.random.permutation(inputs.shape[0])
    train_inputs = inputs[permutation[:-num_eval]]
    train_targets = targets[permutation[:-num_eval]]
    eval_inputs = inputs[permutation[-num_eval:]]
    eval_targets = targets[permutation[-num_eval:]]

    dynamics.input_scaler.fit(train_inputs)
    return train_inputs, train_targets, eval_inputs, eval_targets

def get_random_index(batch_size, ensemble_dim, bootstrap=True):
    if bootstrap:
        return np.stack([np.random.choice(np.arange(batch_size), batch_size, replace=False) for _ in range(ensemble_dim)]).T
    else:
        idx = np.random.choice(np.arange(batch_size), batch_size, replace=False)
        return np.stack([idx for _ in range(ensemble_dim)]).T

def termination_condition(
    dynamics: EnsembleDynamics, 
    stats_epoch: dict, 
    best_eval: list, 
    best_params_list: list,
    epoch_since_last_update: int,
    improvement_ratio: float = 0.01,
    max_epoch_since_update: int = 10,
):
    updated = False
    current_params_list = parse_ensemble_params(dynamics)
    for m in range(dynamics.ensemble_dim):
        current_eval = stats_epoch[f"mae_{m}"]
        improvement = (best_eval[m] - current_eval) / (best_eval[m] + 1e-6)
        if improvement > improvement_ratio:
            best_eval[m] = min(best_eval[m], current_eval)
            best_params_list[m] = current_params_list[m]
            updated = True

    if updated:
        epoch_since_last_update = 0
    else:
        epoch_since_last_update += 1
        
    is_terminate = epoch_since_last_update > max_epoch_since_update
    return is_terminate, best_eval, best_params_list, epoch_since_last_update

def train_ensemble(
    data: TensorDict, 
    pred_rwd: bool,
    dynamics: EnsembleDynamics, 
    optimizer: torch.optim.Optimizer, 
    eval_ratio: float, 
    batch_size: int, 
    epochs: int, 
    bootstrap: bool = True, 
    grad_clip: float | None = None, 
    update_elites: bool = True, 
    max_eval_num: int = 1000, 
    improvement_ratio: float = 0.01,
    max_epoch_since_update: int = 10, 
    logger: Logger | None = None,
    global_step: int | None = None,
):
    train_inputs, train_targets, eval_inputs, eval_targets = format_samples_for_training(
        data, dynamics, pred_rwd, eval_ratio, max_eval_num,
    )
    
    ensemble_dim = dynamics.ensemble_dim
    best_eval = [1e6] * ensemble_dim
    best_params_list = parse_ensemble_params(dynamics)
    epoch_since_last_update = 0
    bar = tqdm(range(epochs))
    for e in bar:
        # shuffle train data
        idx_train = get_random_index(train_inputs.shape[0], ensemble_dim, bootstrap=bootstrap)
        
        train_stats_epoch = []
        for i in range(0, train_inputs.shape[0], batch_size):
            idx_batch = idx_train[i:i+batch_size]
            if len(idx_batch) < batch_size: # drop last batch
                continue

            inputs_batch = train_inputs[idx_batch]
            targets_batch = train_targets[idx_batch]

            loss = dynamics.compute_loss(inputs_batch, targets_batch)

            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(dynamics.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            stats = {"loss": loss.detach().cpu().item()}
            train_stats_epoch.append(stats)
        train_stats_epoch = tree_map(lambda *x: np.stack(x).mean(0), *train_stats_epoch)
        
        # evaluate
        eval_stats_epoch = dynamics.evaluate(eval_inputs, eval_targets)

        # log stats
        stats_epoch = {**train_stats_epoch, **eval_stats_epoch}
        if logger is not None:
            for metric_name, metric_value in stats_epoch.items():
                logger.log_scalar(f"dynamics/{metric_name}", metric_value, global_step)
        
        bar.set_description("ensemble loss: {:.4f}, mae: {:.4f}, terminate: {}/{}".format(
            stats_epoch["loss"], 
            stats_epoch["mae"],
            epoch_since_last_update,
            max_epoch_since_update,
        ))

        (
            is_terminate, 
            best_eval, 
            best_params_list, 
            epoch_since_last_update,
        ) = termination_condition(
            dynamics,
            stats_epoch,
            best_eval,
            best_params_list,
            epoch_since_last_update,
            improvement_ratio,
            max_epoch_since_update,
        )
        if is_terminate:
            break
    
    bar.close()
    set_ensemble_params(dynamics, best_params_list)
    if update_elites:
        dynamics.update_topk_dist(eval_stats_epoch)
    return stats_epoch