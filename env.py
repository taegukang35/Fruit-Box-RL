import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
from ray.rllib.utils import check_env

from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune
config = PPOConfig()

class AppleEnv(gym.Env):
  def __init__(self, render_mode=None, size=(9, 18), time_limit=90):
    self.size = size
    self.time_limit = time_limit
    self.board = None
    self.start_time = None

    self.action_space = spaces.Dict({
        'x_top': spaces.Discrete(n=self.size[0]),
        'y_top': spaces.Discrete(n=size[1]),
        'x_bottom': spaces.Discrete(n=size[0]),
        'y_bottom': spaces.Discrete(n=size[1])
    })

    self.observation_space = spaces.Box(low=0, high=9, shape=self.size, dtype=np.int32)

  def reset(self, seed=None, options=None):
    self.board = self.observation_space.sample()
    zero_idx = np.argwhere(self.board == 0)
    for row, col in zero_idx:
      self.board[row, col] = np.random.randint(1, 10)

    self.start_time = time.time()
    return self.board, {}


  def step(self, action):
    if time.time() - self.start_time > self.time_limit:
      done = True
    else:
      done = False

    if not (action["x_top"] < action["x_bottom"] and action["y_top"] < action["y_bottom"]):
      return self.board, 0, done, False, {}

    reward = 0

    x_top = action["x_top"]
    y_top = action["y_top"]
    x_bottom = action["x_bottom"]
    y_bottom = action["y_bottom"]

    if np.sum(self.board[x_top:x_bottom + 1, y_top:y_bottom + 1]) == 10:
      # count number of nonzero
      reward += np.count_nonzero(self.board[x_top:x_bottom + 1, y_top:y_bottom + 1])
      # change matrix to zero
      self.board[x_top:x_bottom + 1, y_top:y_bottom + 1] = 0

    return self.board, reward, done, False, {}

env = AppleEnv(time_limit=5)

# check_env(env)

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
    ),
    param_space={
        "env": env,
        "kl_coeff": 1.0,
        "num_workers": 1,
        "num_cpus": 1,  # number of CPUs to use per trial
        "num_gpus": 0,  # number of GPUs to use per trial
        # These params are tuned from a fixed starting value.
        "lambda": 0.99,
        "clip_param": 0.2,
        "lr": 1e-4,
    },
    run_config=air.RunConfig(stop={"training_iteration": 1000}),
)
results = tuner.fit()