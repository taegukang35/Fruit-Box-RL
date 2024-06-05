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
        self.sum_matrix = None
        self.legal_actions = []

        self.observation_space = spaces.Box(low=0, high=9, shape=self.size, dtype=int)
        self.action_space = spaces.Dict({
            'x_top': spaces.Discrete(n=size[0]),
            'y_top': spaces.Discrete(n=size[1]),
            'x_bottom': spaces.Discrete(n=size[0], start=1),
            'y_bottom': spaces.Discrete(n=size[1], start=1)
        })

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.board = np.random.choice(np.arange(1, 10), size=self.size)

        self.start_time = time.process_time()
        self.elapsed_time = 0
        
        # this is for obtain available actions
        self.sum_matrix = np.zeros((self.size[0] + 1, self.size[1] + 1), dtype=int)
        for i in range(1, self.size[0] + 1):
            for j in range(1, self.size[1] + 1):
                self.sum_matrix[i, j] = (
                    self.board[i - 1, j - 1] +
                    self.sum_matrix[i - 1, j] +
                    self.sum_matrix[i, j - 1] -
                    self.sum_matrix[i - 1, j - 1]
                )
        
        return self.board, {}
    
    def get_sum(self, x_top, y_top, x_bottom, y_bottom):
        total = (
                self.sum_matrix[x_bottom, y_bottom] 
                - self.sum_matrix[x_top, y_bottom] 
                - self.sum_matrix[x_bottom, y_top] 
                + self.sum_matrix[x_top, y_top]
                )
        return total

    def step(self, action):
        # Update the elapsed time
        done = False
        """self.elapsed_time = time.process_time() - self.start_time
        if self.elapsed_time >= self.time_limit:
            done = True"""
        #if len(self.get_legal_actions()) == 0:
        #    done = True
        if not (action["x_top"] < action["x_bottom"] and action["y_top"] < action["y_bottom"]):
            return self.board, 0, done, False, {}

        reward = 0
        x_top = action["x_top"]
        y_top = action["y_top"]
        x_bottom = action["x_bottom"]
        y_bottom = action["y_bottom"]
        total = self.get_sum(x_top=x_top, y_top=y_top, x_bottom=x_bottom, y_bottom=y_bottom)
        if total == 10:
            reward += np.count_nonzero(self.board[x_top:x_bottom, y_top:y_bottom])
            self.board[x_top:x_bottom, y_top:y_bottom] = 0
        return self.board, reward, done, False, {}
    
        
    def get_legal_actions(self):
        actions = []
        
        self.sum_matrix = np.zeros((self.size[0] + 1, self.size[1] + 1), dtype=int)
        for i in range(1, self.size[0] + 1):
            for j in range(1, self.size[1] + 1):
                self.sum_matrix[i, j] = (
                    self.board[i - 1, j - 1] +
                    self.sum_matrix[i - 1, j] +
                    self.sum_matrix[i, j - 1] -
                    self.sum_matrix[i - 1, j - 1]
                )

        for x_top in range(self.size[0]):
            for y_top in range(self.size[1]):
                for x_bottom in range(x_top + 1, self.size[0] + 1):
                    for y_bottom in range(y_top + 1, self.size[1] + 1):
                        total = self.get_sum(x_top=x_top, y_top=y_top, x_bottom=x_bottom, y_bottom=y_bottom)
                        if total == 10:
                            if self.get_sum(x_top, y_top, x_top+1, y_bottom) == 0:
                                continue
                            if self.get_sum(x_bottom-1, y_top,x_bottom, y_bottom) == 0:
                                continue
                            if self.get_sum(x_top, y_top, x_bottom, y_top+1) == 0:
                                continue
                            if self.get_sum(x_top, y_bottom-1, x_bottom, y_bottom) == 0:
                                continue
                            action = {
                                "x_top": x_top,
                                "y_top": y_top,
                                "x_bottom": x_bottom,
                                "y_bottom": y_bottom
                            }
                            actions.append(action)
                        
                        if total >= 10:
                            break
        return actions


    def render(self, clear=True):
        if clear:
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