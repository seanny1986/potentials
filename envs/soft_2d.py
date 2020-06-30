import numpy as np
import math
from math import sin, cos, tan, radians, degrees, acos, sqrt, pi, exp, copysign
from pygame.math import Vector2
import envs.traj_2d as traj_2d
import gym

import pygame
import pygame.gfxdraw

class SoftTrajectoryEnv2D(traj_2d.TrajectoryEnv2D):
    def __init__(self):
        super(SoftTrajectoryEnv2D, self).__init__()
        
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