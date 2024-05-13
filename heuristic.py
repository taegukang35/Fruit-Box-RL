from env import AppleEnv

env = AppleEnv(time_limit=0.01)
board, done = env.reset()

G = 0

def bruteforce(n):
    global G, done
    for i in range(env.size[0]):
        for j in range(env.size[1]):
            for a in range(i + 1, min(i + n, env.size[0] + 1)):
                for b in range(j + 1,  min(j + n, env.size[1] + 1)):
                    action = {'x_top':i, 'x_bottom':a, 'y_top':j, 'y_bottom':b}
                    board, reward, done, _, _  = env.step(action)
                    if reward > 0:
                        # print(board)
                        G += reward 
eps1 = 47
eps2 = 83
result = []
for _ in range(100):
    while not done:
        if G < eps1:
            bruteforce(4)
        elif eps1 <= G < eps2:
            bruteforce(6)
        else:
            bruteforce(max(env.size))
    result.append(G)
    board, done = env.reset()
    G = 0
print(sum(result)/len(result))