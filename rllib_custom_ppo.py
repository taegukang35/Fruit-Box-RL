from env import AppleEnv
from cnn_net import CustomNetwork
from mlp_net import FullyConnectedNetwork
from mask_net import MaskNetwork
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray import air
from ray import tune
from ray.tune.logger import pretty_print

import gymnasium as gym
from gymnasium import spaces
import numpy as np

ModelCatalog.register_custom_model("custom", CustomNetwork)
register_env("AppleEnv", lambda config: AppleEnv(config))

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        metric="env_runners/episode_reward_mean",
    ),
    param_space={
        "env": "AppleEnv",
        "env_config": {
            "time_limit": 10
        },
        "kl_coeff": 1.0,
        "num_workers": 1,
        "num_cpus": 4,  # number of CPUs to use per trial
        "num_gpus": 0,  # number of GPUs to use per trial
        # These params are tuned from a fixed starting value.
        "lambda": 0.99,
        "clip_param": 0.2,
        "lr": 1e-4,
        "model": {
            "custom_model": "custom",
            "conv_filters": [[4, [4, 4], 1], [16, [8, 8], 1], [32, [4, 4], 1], [1, [4, 4], 1]],
            "fcnet_hiddens": [256, 256]
        }
    },
    run_config=air.RunConfig(stop={"training_iteration": 100}),
)
results = tuner.fit()