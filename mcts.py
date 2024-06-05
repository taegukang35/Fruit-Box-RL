import numpy as np
from env import AppleEnv
import random
import time
import math
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

class TreeNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.score = 0
        self.visits = 0
        self.action = action  # prev action for this state
        self.children = {}  # key: board 상태, value: board 상태에 대응되는 TreeNode

class GameState:
    def __init__(self, seed=None):
        self.env = AppleEnv()
        self.env.reset(seed=seed)
        self.board = self.env.board
        self.total_rewards = 0
        self.depth = 0
        
    def clone(self):
        clone_state = GameState()
        clone_state.board = self.board
        clone_state.total_rewards = self.total_rewards
        clone_state.depth = self.depth
        return clone_state

    def act(self, action):
        self.board, reward, done, truncated, _ = self.env.step(action)
        self.total_rewards += reward
        self.depth += 1
        return self.board, reward, done, truncated, {}
    
    def get_legal_actions(self):
        return self.env.get_legal_actions()

    def is_terminal(self):
        return len(self.env.get_legal_actions()) == 0

    def print_board(self):
        self.env.render()
        
def hash_string_md5(input_string):
    encoded_string = input_string.encode('utf-8')
    md5_hash = hashlib.md5()
    md5_hash.update(encoded_string)
    return md5_hash.hexdigest()

class MCTS:
    def __init__(self, max_iterations=1, max_depth=10, num_rollouts=10):
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.num_rollouts = num_rollouts

    def search(self, init_state: GameState, criteria="avg"):
        if init_state.is_terminal():
            return -1

        rootNode = TreeNode(init_state)
        self.add_child(rootNode)

        for _ in tqdm.tqdm(range(self.max_iterations)):
            current = rootNode
            while len(current.children) != 0:
                current = self.select(current, criteria="ucb")

            if current.visits == 0:
                rewards = self.rollout(current)
                for reward in rewards:
                    self.backpropagate(current, reward)
            else:
                if not current.state.is_terminal():
                    self.add_child(current)
                    _, current = list(current.children.items())[0]
                rewards = self.rollout(current)
                for reward in rewards:
                    self.backpropagate(current, reward)

        best_child = self.select(rootNode, criteria=criteria)
        return best_child.action

    def add_child(self, node):
        actions = node.state.get_legal_actions()
        for action in actions:
            new_state = node.state.clone()
            new_state.act(action)
            if hash_string_md5(str(new_state.board)) not in node.children:
                new_node = TreeNode(new_state, parent=node, action=action)
                node.children[hash_string_md5(str(new_state.board))] = new_node

    def select(self, node, criteria="ucb"):
        max_key = None
        max_ucb = -float('inf')
        
        if criteria == "ucb":
            for key, child in node.children.items():
                ucb_val = float('inf') if child.visits == 0 else (
                    child.score / child.visits +
                    (math.sqrt(math.log(node.visits) / child.visits) * 1.41)
                )

                if ucb_val > max_ucb:
                    max_ucb = ucb_val
                    max_key = key
                    
        elif criteria == "visit":
            for key, child in node.children.items():
                ucb_val = node.visits
                if ucb_val > max_ucb:
                    max_ucb = ucb_val
                    max_key = key
        else: # avg
            for key, child in node.children.items():
                ucb_val = float('inf') if child.visits == 0 else child.score / child.visits

                if ucb_val > max_ucb:
                    max_ucb = ucb_val
                    max_key = key

        return node.children[max_key]

    def rollout_single(self, state, seed):
        rng = random.Random(seed)
        current_state = state.clone()
        while current_state.depth <= self.max_depth:
            actions = current_state.get_legal_actions()
            if len(actions) == 0:
                break
            action = random.choice(actions)
            current_state.act(action)
        return current_state.total_rewards

    def rollout(self, node):
        seeds = [random.randint(0, 1_000_000) for _ in range(self.num_rollouts)]
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.rollout_single, node.state, seed) for seed in seeds]
            results = [future.result() for future in as_completed(futures)]
        return results

    def backpropagate(self, node, reward):
        while node.parent is not None:
            node.visits += 1
            node.score += reward
            node = node.parent
        node.visits += 1

num_worker = [1, 16, 32]
max_iter = [100, 200, 500, 1000]
criteria = ["visit", "avg"]

def main():
    for c in criteria:
        for i in max_iter:
            for n in num_worker:
                state = GameState(seed=1)
                while not state.is_terminal():
                    # state.print_board()
                    mcts = MCTS(max_iterations=i, max_depth=10000, num_rollouts=n)
                    action = mcts.search(state, criteria=c)
                    state.act(action)
                    
                # state.print_board()
                print(f"criteria: {c}, num_worker: {n}, num_iter: {i}, rewards: {state.total_rewards}")

if __name__ == "__main__":
    main()