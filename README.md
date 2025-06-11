<div align="center">
  <img width="300px" height="auto" src="assets/logo.png">
</div>

# CleanIL (Clean Implementation of IL Algorithms)
CleanIL is a deep imitation and inverse reinforcement learning library. Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl), we provide single-file implementations of SOTA algorithms. We use [TorchRL](https://github.com/pytorch/rl) for some utility functions like replay buffers without hurting code readability. Feel free to read the [release blog post](https://latentobservations.substack.com/p/introducing-cleanil-for-imitation).

## Getting started

### Installation
Clone the repo. Create a conda environment with Python >= 3.10. Then install the dependencies.
```
git clone https://github.com/ran-weii/cleanil.git
conda create -n cleanil python=3.11
conda activate cleanil
pip install -r requirements.txt
pip install -e .
```

### Usage
Implemented algorithms are under `cleanil/il`. The configuration files and scripts for running the algorithms are in `configs` and `scripts`, respectively. In the configs, you can choose to log with `wandb`, `csv`, or other methods supported by `torchrl`. 

To get started, run the seminal GAIL as follows:
```
cd scripts/il
sh train_gail.sh
```

For offline model-based algorithms, you should train the dynamics model first before running the IL scripts. You can use the following:
```
cd scripts/dynamics
sh train_ensemble.sh
```
Alternatively, you can download pretrained dynamics models from this [HuggingFace collection](https://huggingface.co/collections/ran-w/d4rl-mujoco-dynamics-models-67cbb2991b69d63fac266d7a).

We have also implemented a set of RL algorithms under `cleanil/rl`. The training scripts are in `scripts/rl`. Although we have tested some of them to ensure the base RL solvers for the IL algorithms are performant, the tests were not comprehensive.

## Algorithms and benchmarks
The implemented IL and IRL algorithms are listed in the following table. For some of the algorithms, we have implemented modern techniques and tricks such as gradient penalty that may deviate slightly from the ideal algorithms in the papers. For behavior cloning, we do not use any regularization by default.

We use 20 expert trajectories for behavior cloning based methods and 10 for RL based methods. For algorithms that make use of an offline transition dataset (e.g., from [D4RL](https://github.com/Farama-Foundation/D4RL)), we choose the `medium-replay` dataset. This dataset is more challenging as it does not contain expert transitions and is much smaller in size (way less than 1M transitions).

The numbers below are the inter-quartile means and standard deviations of normalized returns from the last 30 evaluation episodes of a single seed. We did not cherry pick seeds, all algorithms were run on seed 0. The Wandb run logs are linked under the scores. 

We also documented some implementation tricks and observations. See this [blog post](https://ran-weii.github.io/2025/03/28/cleanil-implementation-tricks.html) for detail.

| Paper | Algorithm | On/offline | Num expert traj. | Halfcheetah | Hopper | Walker2d |
|-------|-----------|------------|------------------|-------------|--------|----------|
| Behavior Cloning | [bc](cleanil/il/bc.py) | Off | 20 | [46.24 ± 19.48](https://wandb.ai/ranw/cleanil_bc_halfcheetah_benchmark?nw=nwuserranw) | [86.27 ± 11.65](https://wandb.ai/ranw/cleanil_bc_hopper_benchmark?nw=nwuserranw) | [99.54 ± 0.17](https://wandb.ai/ranw/cleanil_bc_walker2d_benchmark?nw=nwuserranw) |
| [Implicit Behavioral Cloning](https://arxiv.org/abs/2109.00137) | [ibc](cleanil/il/ibc.py) | Off | 20 | [48.07 ± 18.38](https://wandb.ai/ranw/cleanil_ibc_halfcheetah_benchmark?nw=nwuserranw) | [69.07 ± 17.97](https://wandb.ai/ranw/cleanil_ibc_hopper_benchmark?nw=nwuserranw) | [73.33 ± 35.71](https://wandb.ai/ranw/cleanil_ibc_walker2d_benchmark?nw=nwuserranw) |
| [IQ-Learn: Inverse soft-Q Learning for Imitation](https://arxiv.org/abs/2106.12142) | [iqlearn](cleanil/il/iqlearn.py) | Off | 20 | [10.66 ± 3.81](https://wandb.ai/ranw/cleanil_iqlearn_halfcheetah_benchmark?nw=nwuserranw) | [84.23 ± 13.91](https://wandb.ai/ranw/cleanil_iqlearn_hopper_benchmark?nw=nwuserranw) | [81.23 ± 31.29](https://wandb.ai/ranw/cleanil_iqlearn_walker2d_benchmark?nw=nwuserranw) |
| [Dual RL: Unification and New Methods for Reinforcement and Imitation Learning](https://arxiv.org/abs/2302.08560) | [recoil](cleanil/il/recoil.py) | Off | 20 | [70.70 ± 9.31](https://wandb.ai/ranw/cleanil_recoil_halfcheetah_benchmark?nw=nwuserranw) | [86.51 ± 15.32](https://wandb.ai/ranw/cleanil_recoil_hopper_benchmark?nw=nwuserranw) | [99.55 ± 0.30](https://wandb.ai/ranw/cleanil_recoil_walker2d_benchmark?nw=nwuserranw) |
| [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) | [gail](cleanil/il/gail.py) | On | 10 | [107.26 ± 0.85](https://wandb.ai/ranw/cleanil_gail_halfcheetah_benchmark?nw=nwuserranw) | [101.10 ± 0.27](https://wandb.ai/ranw/cleanil_gail_hopper_benchmark?nw=nwuserranw) | [100.75 ± 0.22](https://wandb.ai/ranw/cleanil_gail_walker2d_benchmark?nw=nwuserranw) |
| [When Demonstrations Meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning](https://arxiv.org/abs/2302.07457) | [omlirl](cleanil/il/omlirl.py) | Off | 10 | [96.51 ± 1.42](https://wandb.ai/ranw/cleanil_omlirl_halfcheetah_benchmark?nw=nwuserranw) | [99.91 ± 0.27](https://wandb.ai/ranw/cleanil_omlirl_hopper_benchmark?nw=nwuserranw) | [99.34 ± 0.49](https://wandb.ai/ranw/cleanil_omlirl_walker2d_benchmark?nw=nwuserranw) |
| [A Bayesian Approach to Robust Inverse Reinforcement Learning](https://arxiv.org/abs/2309.08571) | [rmirl](cleanil/il/rmirl.py) | Off | 10 | [99.09 ± 2.52](https://wandb.ai/ranw/cleanil_rmirl_halfcheetah_benchmark?nw=nwuserranw) | [100.28 ± 0.19](https://wandb.ai/ranw/cleanil_rmirl_hopper_benchmark?nw=nwuserranw) | [96.44 ± 1.03](https://wandb.ai/ranw/cleanil_rmirl_walker2d_benchmark?nw=nwuserranw) |

## Known issues and limitations
* Currently, all experiments are conducted on cpu from a MacBook Pro. Running RL with torch MPS is known to be slow (see [this issue](https://github.com/pytorch-labs/LeanRL/issues/16)). But we also do not have access to CUDA so did not test the code with CUDA.
* [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) notably increased the speed of the algorithms, but optimal usage was not explored.

## Contributing and getting involved
Feel free to post any issues or submit pull requests. Contributing algorithms are also wellcome; please post an issue about it. 

## Acknowledgements
* [CleanRL](https://github.com/vwxyzjn/cleanrl): for inspiring our single-file implementations
* [TorchRL](https://github.com/pytorch/rl): for building convenient RL utilities.