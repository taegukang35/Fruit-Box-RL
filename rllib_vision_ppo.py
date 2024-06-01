from env import AppleEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        metric="env_runners/episode_reward_mean",
    ),
    param_space={
        "env": AppleEnv,
        "env_config": {"time_limit": 10},
        "kl_coeff": 1.0,
        "num_workers": 1,
        "num_cpus": 8,  # number of CPUs to use per trial
        "num_gpus": 0,  # number of GPUs to use per trial
        # These params are tuned from a fixed starting value.
        "lambda": 0.99,
        "clip_param": 0.2,
        "lr": 1e-4,
        "model": {
            "conv_filters": [[2, 2, 1], [4, 4, 1], [8, 8, 1], [1, 8, 1]],
            "post_fcnet_hiddens": [256, 256]
        }
    },
    run_config=air.RunConfig(stop={"training_iteration": 10000}),
)
results = tuner.fit()