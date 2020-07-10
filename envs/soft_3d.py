import gym
import gym_aero
import envs.traj_3d as trajectory_env
import random
from math import sin, cos, exp
from gym import spaces
import numpy as np

class SoftTrajectoryEnv3D(trajectory_env.TrajectoryEnv3D):
    def __init__(self):
        super(SoftTrajectoryEnv3D, self).__init__()

    def switch_goal(self, data):
        dist = self.pdf_norm * exp(-self.temperature * self.curr_dist ** 2)
        sample = np.random.RandomState().uniform(low=0, high=self.pdf_norm)
        if sample <= dist: return True
        else: return False
    
    def term_reward(self, term):
        if term:
            return 100 * exp(-self.temperature* self.curr_dist**2)
        else:
            return 0