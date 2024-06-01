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
        "kl_coeff": 1.0,
        "num_workers": 1,
        "num_cpus": 32,  # number of CPUs to use per trial
        "num_gpus": 0,  # number of GPUs to use per trial
        # These params are tuned from a fixed starting value.
        "lambda": 0.99,
        "clip_param": 0.2,
        "lr": 1e-4,
    },
    run_config=air.RunConfig(stop={"training_iteration": 10000}),
)
results = tuner.fit()