import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import os
            
class AppleEnv(gym.Env):
    def __init__(self, render_mode=None, size=(9, 18), time_limit=90):
        self.size = size
        self.time_limit = time_limit
        self.board = None
        self.start_time = None
        self.elapsed_time = 0

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

        self.start_time = time.process_time()
        self.elapsed_time = 0
        return self.board, {}


    def step(self, action):
        # Update the elapsed time
        done = False
        self.elapsed_time = time.process_time() - self.start_time
        if self.elapsed_time >= self.time_limit:
            print("Time limit exceeded!")
            done = True
        # if len(self.get_legal_actions()) == 0:
        #     done = True
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

    # brute force
    def get_legal_actions(self):
        actions = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for a in range(i + 1, self.size[0] + 1):
                    for b in range(j + 1, self.size[1] + 1):
                        action = {'x_top':i, 'x_bottom':a, 'y_top':j, 'y_bottom':b}
                        x_top = action["x_top"]
                        y_top = action["y_top"]
                        x_bottom = action["x_bottom"]
                        y_bottom = action["y_bottom"]
                        if np.sum(self.board[x_top:x_bottom, y_top:y_bottom]) == 10:
                            actions.append(action)
        return actions
    
    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("   " + " ".join(f"{i:2}" for i in range(self.size[1])))
        print("  +" + "---" * self.size[1] + "+")

        for i in range(self.size[0]):
            row_str = f"{i:2}|"
            for j in range(self.size[1]):
                cell_value = self.board[i, j]
                cell_str = f"{cell_value:2}" if cell_value != 0 else " ."
                row_str += f" {cell_str}"
            row_str += " |"
            print(row_str)
        
        print("  +" + "---" * self.size[1] + "+")