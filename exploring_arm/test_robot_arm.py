import gym
import numpy as np
import time
import real_robots
from policy import RandomPolicy
from real_robots.envs import REALRobotEnv
from wrapper import WrapRobot

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2


print('setting up environment')
#env = gym.make("REALRobot2020-R2J3-v0")
env = REALRobotEnv(objects=1)
env = WrapRobot(env, crop_obs=True)

print('setting up ppo model')
model = PPO2(CnnPolicy, env, verbose=1)
print('start learning')
model.learn(total_timesteps=256)
print('learning done')

#Here we need to restart the environent to make rendering possible
env = REALRobotEnv(objects=1)
env = WrapRobot(env, crop_obs=True)

env.render("human")

print('display model')
observation = env.reset()
action = env.action_space.sample()
reward, done = 0, False
for t in range(400):
    model_action,_,_,_ = model.step([observation])#actions, values, self.states, neglogpacs

    observation, reward, done, info = env.step(model_action[0])
