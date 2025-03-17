"""
Adapted from torchrl https://github.com/pytorch/rl/tree/main
"""
import functools
from dataclasses import dataclass
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter


@dataclass
class EnvConfig:
    name: str = "HalfCheetah-v4"
    task: str = ""
    library: str = "gymnasium"
    max_episode_steps: int = 1000
    num_workers: int = 1 # number of parallel rollout workers


def env_maker(config: EnvConfig, device="cpu", from_pixels=False):
    lib = config.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                config.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            config.name, config.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")

def apply_env_transforms(env, max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env

"""TODO allow video logging"""
def make_environment(config: EnvConfig, device="cpu", logger=None, seed=0):
    """Make environments for training and evaluation."""
    partial = functools.partial(env_maker, config=config, device=device)
    parallel_env = ParallelEnv(
        config.num_workers,
        EnvCreator(partial),
        serial_for_single=True,
    )
    parallel_env.set_seed(seed)

    train_env = apply_env_transforms(parallel_env, config.max_episode_steps)

    # partial = functools.partial(env_maker, config=config, from_pixels=config.logger.video)
    partial = functools.partial(env_maker, config=config)
    trsf_clone = train_env.transform.clone()
    # if config.logger.video:
    #     trsf_clone.insert(
    #         0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
    #     )
    eval_env = TransformedEnv(
        ParallelEnv(
            config.num_workers,
            EnvCreator(partial),
            serial_for_single=True,
        ),
        trsf_clone,
    )
    return train_env, eval_env