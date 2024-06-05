from env import AppleEnv
import time
import random
import numpy as np
np.random.seed(42)

env = AppleEnv()
board, done = env.reset()
# env.render()

G = 0
done = False
    
while not done:
    actions = env.get_legal_actions()
    # print(actions)
    # break
    if len(actions) == 0:
        break
    action = random.choice(env.get_legal_actions())
    #print(action)
    reward, done = env.step(action)[1:3]
    G += reward
    env.render()
    #print(env.sum_matrix)
    #break
print(G)