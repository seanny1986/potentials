import numpy as np
import math
from math import sin, cos, tan, radians, degrees, acos, sqrt, pi, exp, copysign
from pygame.math import Vector2
import envs.traj_2d as traj_2d
import gym

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

    def reset(self):
        self.t = 0
        self.goal_counter = 0
        self.goal_list_xy = []
        xy_ = np.array([1.5, 0])
        self.goal_list_xy.append(xy_.copy())
        for _ in range(self.traj_len-1):
            angle = np.random.RandomState().uniform(low=-pi/2, high=pi/2)
            temp = np.array([1.5*cos(angle), 1.5*sin(angle)])
            xy_ += temp
            self.goal_list_xy.append(xy_.copy())

        self.goal_list_zeta = []
        self.goal_list_uv = []
        self.goal_list_r = []
        for i in range(self.traj_len):
            self.goal_list_zeta.append(0.)
            self.goal_list_uv.append([0., 0.])
            self.goal_list_r.append(0.)
        
        xy, zeta, uv, r = self.player.reset()
        angle = np.random.RandomState().uniform(low=-pi/2, high=pi/2)
        self.player.angle = degrees(zeta)
        sin_zeta, cos_zeta = sin(zeta), cos(zeta)
        self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), np.zeros((2,)))
        self.set_prev_dists()
        self.obs = self.get_obs((xy, sin_zeta, cos_zeta, uv, r), np.zeros((2,)))
        return self.obs
