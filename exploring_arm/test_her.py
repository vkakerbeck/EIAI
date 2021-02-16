import gym
import numpy as np
import time
import real_robots
from real_robots.envs import REALRobotEnv
from wrapper import GoalWrapper

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import HER, SAC


print('setting up environment')
#env = gym.make("REALRobot2020-R2J3-v0")
env = REALRobotEnv(objects=1)
# Currently this wrapper doesn't really return goals but just sample_placeholder
# to match the her format.
env = GoalWrapper(env, crop_obs=True)

print('setting up model')
model = HER('MlpPolicy', env, SAC, n_sampled_goal=4,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256)
print('start learning')
model.learn(total_timesteps=256)
print('learning done')

#Here we need to restart the environent to make rendering possible
#(doesn't work with the wrappers right now)
env = REALRobotEnv(objects=1)
env = GoalWrapper(env, crop_obs=True)
env.render("human")

print('display model')
observation = env.reset()
action = env.action_space.sample()
reward, done = 0, False
for t in range(100):
    model_action,_ = model.predict(observation)

    observation, reward, done, info = env.step(model_action)
    #print(model_action)
