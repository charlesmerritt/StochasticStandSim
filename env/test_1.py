from ForestEnv import ForestStandEnv

env = ForestStandEnv()
state, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(action)
    print(state, reward)
