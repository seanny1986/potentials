import numpy as np
import math
from math import sin, cos, tan, radians, degrees, acos, sqrt, pi, exp, copysign
from pygame.math import Vector2
import envs.fan_2d as fan_2d
import gym

import pygame
import pygame.gfxdraw

class SoftFanTrajectoryEnv2D(fan_2d.FanTrajectoryEnv2D):
    def __init__(self):
        super(SoftFanTrajectoryEnv2D, self).__init__()
        
    def switch_goal(self, data):
        dist = self.pdf_norm * exp(-self.temperature * self.curr_dist ** 2)
        sample = np.random.RandomState().uniform(low=0, high=self.pdf_norm)
        if sample <= dist: return True
        else: return False