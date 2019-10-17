import numpy as np
from gym import spaces

from math import sin, cos
from pygame.math import Vector2

import pygame
import pygame.gfxdraw

import envs.traj_2d as traj_2d

class WaypointEnv2D(traj_2d.TrajectoryEnv2D):
    def __init__(self, dt=0.05):
        super(WaypointEnv2D, self).__init__(dt)
        self.num_fut_wp = 0
        self.traj_len = 1
        state_size = 10+7*(self.num_fut_wp+1)
        self.observation_space = spaces.Box(-1, 1, shape=(state_size,))