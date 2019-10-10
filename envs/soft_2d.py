import numpy as np
import math
from math import sin, cos, tan, radians, degrees, acos, sqrt, pi, exp, copysign
from pygame.math import Vector2
import envs.traj_2d as traj_2d

import pygame
import pygame.gfxdraw

class TrajectoryEnv2D(traj_2d.TrajectoryEnv2D):
    def __init__(self, dt=0.05):
        super(TrajectoryEnv2D, self).__init__(dt)

    def switch_goal(self, state):
        xy, sin_zeta, cos_zeta, uv, r = state
        u = self.curr_dist
        if self.goal_counter <= self.traj_len-2:
            v = sum([(x-g)**2 for x, g in zip(xy, self.goal_list_xy[self.goal_counter+1])])**0.5
        else:
            v = 0
        dist = exp(-self.temperature*(u**2+u*v))
        sample = np.random.RandomState().uniform(low=0, high=1)
        if sample < dist: return True
        else: return False