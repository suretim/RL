#import gymnasium as gym
from plantnanny_env import PlantNannyEnv

env = PlantNannyEnv()
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    #print(info,"terminated=",terminated,"truncated=",truncated)
    print(info)
    env.render()
    if terminated or truncated:
        break
