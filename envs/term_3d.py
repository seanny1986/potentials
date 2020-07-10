import gym
import gym_aero
import envs.traj_3d as tenv
import random
from math import sin, cos, exp, log, pi
from gym import spaces
import numpy as np

class TrajectoryEnvTerm3D(tenv.TrajectoryEnv3D):
    def __init__(self):
        super(TrajectoryEnvTerm3D, self).__init__()

    def term_reward(self, term):
        if term: y = 1
        else: y = 0
        lp = -pi * self.curr_dist ** 2
        p = exp(lp)
        rew = y * lp + (1 - y) * log(1 - p + 1e-16)
        return rew

    def switch_goal(self, data):
        term = data[-1]
        if term == 1: return True
        else: return False