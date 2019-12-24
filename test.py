import gym
import longdpole

env = gym.make('LongdpoleEnv-v0')
env.reset()
for _ in range(1000):
    print(env.action_space.sample())
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(observation)
    env.render()
    if (done):
        break
env.close()
