import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time

class AppleEnv(gym.Env):
  def __init__(self, render_mode=None, size=(9, 18), time_limit=90):
    self.size = size
    self.time_limit = time_limit
    self.board = None
    self.start_time = None

    self.action_space = spaces.Dict({
        'x_top': spaces.Discrete(n=size[0]),
        'y_top': spaces.Discrete(n=size[1]),
        'x_bottom': spaces.Discrete(n=size[0], start=1),
        'y_bottom': spaces.Discrete(n=size[1], start=1)
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

    if np.sum(self.board[x_top:x_bottom, y_top:y_bottom]) == 10:
      # count number of nonzero
      reward += np.count_nonzero(self.board[x_top:x_bottom, y_top:y_bottom])
      # change matrix to zero
      self.board[x_top:x_bottom, y_top:y_bottom] = 0

    return self.board, reward, done, False, {}