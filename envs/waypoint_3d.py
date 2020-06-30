import numpy as np
from gym import spaces

from math import sin, cos, pi

import gym
import gym_aero
import envs.traj_3d as trajectory_env

class WaypointEnv3D(trajectory_env.TrajectoryEnv3D):
    def __init__(self):
        super(WaypointEnv3D, self).__init__()
        self.num_fut_wp = 0
        self.traj_len = 1
        state_size = 5+15*(self.num_fut_wp+1)
        self.observation_space = spaces.Box(-1, 1, shape=(state_size,))
    
    def generate_waypoint(self):
        phi = np.random.RandomState().uniform(low=-2*pi, high=2*pi)
        theta = np.random.RandomState().uniform(low=-2*pi, high=2*pi)
        rad = np.random.RandomState().uniform(low=1, high=1.5)
        y = rad*sin(theta)*cos(phi)
        x = rad*cos(theta)*cos(phi)
        z = -rad*sin(theta)
        return [x, y, z]