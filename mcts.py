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
    def __init__(self):
        self.env = AppleEnv()
        self.env.reset()
        self.board = self.env.board
        self.total_rewards = 0
        
    def clone(self):
        clone_state = GameState()
        clone_state.board = self.board
        clone_state.total_rewards = self.total_rewards
        return clone_state

    def act(self, action):
        self.board, reward, done, truncated, _ = self.env.step(action)
        self.total_rewards += reward
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

    def search(self, init_state: GameState):
        if init_state.is_terminal():
            return -1

        rootNode = TreeNode(init_state)
        self.add_child(rootNode)

        for _ in tqdm.tqdm(range(self.max_iterations)):
            current = rootNode
            while len(current.children) != 0:
                current = self.select(current, ucb=True)

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

        best_child = self.select(rootNode, ucb=False)
        return best_child.action

    def add_child(self, node):
        actions = node.state.get_legal_actions()
        for action in actions:
            new_state = node.state.clone()
            new_state.act(action)
            if hash_string_md5(str(new_state.board)) not in node.children:
                new_node = TreeNode(new_state, parent=node, action=action)
                node.children[hash_string_md5(str(new_state.board))] = new_node

    def select(self, node, ucb=True):
        max_ucb = -float('inf')
        max_key = None

        for key, child in node.children.items():
            ucb_val = float('inf') if child.visits == 0 else (
                child.score / child.visits +
                (math.sqrt(math.log(node.visits) / child.visits) * (1.41 if ucb else 0))
            )

            if ucb_val > max_ucb:
                max_ucb = ucb_val
                max_key = key

        return node.children[max_key]

    def rollout_single(self, state):
        current_state = state.clone()
        depth = 0
        while depth <= self.max_depth:
            actions = current_state.get_legal_actions()
            if len(actions) == 0:
                break
            action = random.choice(actions)
            current_state.act(action)
            depth += 1
        return current_state.total_rewards

    def rollout(self, node):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.rollout_single, node.state) for _ in range(self.num_rollouts)]
            results = [future.result() for future in as_completed(futures)]
        return results

    def backpropagate(self, node, reward):
        while node.parent is not None:
            node.visits += 1
            node.score += reward
            node = node.parent
        node.visits += 1

def main():
    state = GameState()

    while not state.is_terminal():
        state.print_board()
        mcts = MCTS(max_iterations=300, max_depth=50, num_rollouts=16)
        action = mcts.search(state)
        state.act(action)
        
    state.print_board()
    print("Total rewards:", state.total_rewards)

if __name__ == "__main__":
    main()
