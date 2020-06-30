import numpy as np
import gym
from gym import spaces

import math
from math import sin, cos, radians, sqrt, acos, exp, log, pi

import pygame
import pygame.gfxdraw
import pygame.image as image
import envs.traj_2d as tenv

class TrajectoryEnvTerm2D(tenv.TrajectoryEnv2D):
    def __init__(self):
        super(TrajectoryEnvTerm2D, self).__init__()

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
