import gym
from gym import spaces
import numpy as np

class WrapRobot(object):
    '''
    Wrapper for the REALRobotEnv (based on MJCFBaseBulletEnv).
    Returns only Box of retina image as observation instead of
    Dictionary of (joint_positions, touch_sensors, retina, depth,
    mask, object_positions, goal, goal_mask, goal_positions).
    This is required for the openAI stable-baselines algorithms
    as they don't support Dict observations.
    Also the action space is reformated into the Box format, omitting
    the render parameter.
    '''
    def __init__(self, env):
        self._env = env

        self._env.action_space = self._env.action_space['joint_command']
        self._env.observation_space = self._env.observation_space['retina']

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        action = {
            'joint_command': action,
            'render': True,
        }
        observation, reward, done, info = self._env.step_joints(action)
        observation = observation['retina']
        return observation, reward, done, info

    def reset(self):
        obs = self._env.reset()
        return obs['retina']

class GoalWrapper(object):
        '''
        Wrapper for the REALRobotEnv (based on MJCFBaseBulletEnv).
        Returns Dict of 3 Box items (observation, achieved_goal,
        desired_goal) as observation instead of the entire
        Dictionary of (joint_positions, touch_sensors, retina, depth,
        mask, object_positions, goal, goal_mask, goal_positions).
        This is required follows the naming and format of the openAI
        stable-baselines her implementation.
        For observation the retina image is used, the two goal elements
        are currently placeholder.
        The stable_baselines implementation currently only supports 1D
        observations which is why the images are flattened.
        Also the action space is reformated into the Box format, omitting
        the render parameter.
        '''
    def __init__(self, env):
        self._env = env

        obs_shape = env.observation_space['retina'].shape
        obs_shape_1D = obs_shape[0]*obs_shape[1]*obs_shape[2]

        self._env.action_space = self._env.action_space['joint_command']
        # TODO: replace goal placeholders with actual goals
        self._env.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(low=0, high=255.0, shape=[obs_shape_1D], dtype='float32'),
            achieved_goal=spaces.Box(low=0, high=255.0, shape=[obs_shape_1D], dtype='float32'),
            observation=spaces.Box(low=0, high=255.0,shape=[obs_shape_1D], dtype='float32')
        ))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        action = {
            'joint_command': action,
            'render': True,
        }
        observation, reward, done, info = self._env.step_joints(action)
        observation = {
            'observation': observation['retina'].flatten(),
            'achieved_goal': observation['goal'].flatten(),
            'desired_goal': observation['goal'].flatten(),
        }
        return observation, reward, done, info
    def reset(self):
        observation = self._env.reset()
        observation = {
            'observation': observation['retina'].flatten(),
            'achieved_goal': observation['goal'].flatten(),
            'desired_goal': observation['goal'].flatten(),
        }
        return observation
