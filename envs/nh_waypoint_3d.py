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
    
    def reward(self, state, action, normalized_rpm):
        _, _, _, uvw, _ = state
        reward, info = super(WaypointEnv3D, self).reward(state, action, normalized_rpm)
        heading_rew = -10*uvw[1]**2
        forward_rew = 10*uvw[0]/abs(uvw[0])/self.curr_dist if self.curr_dist > self.goal_thresh else 10*uvw[0]/abs(uvw[0])/self.goal_thresh
        reward += heading_rew+forward_rew
        return reward, info
    
    def generate_waypoint(self):
        phi = np.random.RandomState().uniform(low=-2*pi, high=2*pi)
        theta = np.random.RandomState().uniform(low=-2*pi, high=2*pi)
        rad = np.random.RandomState().uniform(low=1, high=1.5)
        y = rad*sin(theta)*cos(phi)
        x = rad*cos(theta)*cos(phi)
        z = -rad*sin(theta)
        return [x, y, z]