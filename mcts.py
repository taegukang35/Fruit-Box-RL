import numpy as np
from env import AppleEnv
import random

class TreeNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state # 노드에 대응되는 GameState 저장
        self.parent = parent
        self.score = 0
        self.visits = 0
        self.action = action # prev action for this state
        self.children = {} # key: board 상태, value: board 상태에 대응되는 TreeNode

class GameState:
    def __init__(self):
        self.env = AppleEnv(time_limit=10)
        self.board = self.env.board
        self.env.reset()
        self.legal_actions = self.get_legal_actions()
        
    def clone(self):
        # 게임 상태의 복사본을 반환
        return self.env.board.copy()

    def get_legal_actions(self):
        actions = []
        for i in range(self.env.size[0]):
            for j in range(self.env.size[1]):
                for a in range(i + 1, self.env.size[0] + 1):
                    for b in range(j + 1, self.env.size[1] + 1):
                        action = {'x_top':i, 'x_bottom':a, 'y_top':j, 'y_bottom':b}
                        x_top = action["x_top"]
                        y_top = action["y_top"]
                        x_bottom = action["x_bottom"]
                        y_bottom = action["y_bottom"]
                        if np.sum(self.env.board[x_top:x_bottom, y_top:y_bottom]) == 10:
                            actions.append(action)
        return actions

    def act(self, action):
        # 주어진 행동을 현재 상태에 적용
        self.board, reward, done, truncated, _ = self.env.step(action)
        return self.board, reward, done, truncated, {}

    def is_terminal(self):
        # 게임이 종료되었는지 확인
        return self.get_winner() is not None or self.is_board_full()

    def print_board(self):
        # 현재 보드를 출력
        for row in self.board:
            print("|".join(row))
            print("-----")
        print()

game = GameState()
done = False
rewards = 0
while not done:
    actions = game.get_legal_actions()
    if len(actions) == 0:
        break
    action = random.sample(actions, k=1)[0]
    print(action)
    board, reward, done, truncated, _ = game.act(action)
    rewards += reward
    game.env.render()
print(rewards)