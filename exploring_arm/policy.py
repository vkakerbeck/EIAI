from real_robots.policy import BasePolicy
import numpy as np

class RandomPolicy(BasePolicy):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.action = action_space.sample()

    def step(self, observation, reward, done):
        if np.random.rand() < 0.05:
            self.action = self.action_space.sample()
        return self.action

    """
    add the start_intrinsic_phase, end_intrinsic_phase, start_extrinsic_phase,
    end_extrinsic_phase, start_extrinsic_trial, end_extrinsic_trial functions
    from https://github.com/AIcrowd/real_robots/blob/6dd5b70bad14426483e2d3ee29b3d8708d34e1ba/real_robots/policy.py
    to perform actions at start or end of phases and trials.
    """
