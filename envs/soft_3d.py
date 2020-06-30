import gym
import gym_aero
import envs.traj_3d as trajectory_env
import random
from math import sin, cos, exp
from gym import spaces
import numpy as np

class TrajectoryEnv3D(trajectory_env.TrajectoryEnv3D):
    def __init__(self):
        super(TrajectoryEnv3D, self).__init__()
        self.num_fut_wp = 3
        state_size = (1+self.num_fut_wp)*15+5
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(state_size,))
        self.temperature = 10

    def switch_goal(self, state):
        xyz, sin_zeta, cos_zeta, uvw, pqr = state
        u = self.curr_dist
        if self.goal_counter <= self.traj_len-2:
            v = sum([(x-g)**2 for x, g in zip(xyz, self.goal_list_xyz[self.goal_counter+1])])**0.5
        else:
            v = 0
        dist = exp(-self.temperature*(u**2+u*v))
        sample = np.random.RandomState().uniform(low=0, high=1)
        if sample < dist: return True
        else: return False

    def step(self, data):
        action = data
        commanded_rpm = self.translate_action(action)
        xyz, zeta, uvw, pqr = super(trajectory_env.TrajectoryEnv, self).step(commanded_rpm)
        sin_zeta = [sin(z) for z in zeta]
        cos_zeta = [cos(z) for z in zeta]
        current_rpm = self.get_rpm()
        normalized_rpm = [rpm/self.max_rpm for rpm in current_rpm]
        self.set_curr_dists((xyz, sin_zeta, cos_zeta, uvw, pqr), commanded_rpm, normalized_rpm)
        reward, info = self.reward((xyz, sin_zeta, cos_zeta, uvw, pqr), commanded_rpm, normalized_rpm)
        term = self.switch_goal((xyz, sin_zeta, cos_zeta, uvw, pqr))
        if term:
            if self.goal_counter < self.traj_len-1:
                self.goal_counter += 1
                self.set_current_dists((xyz, sin_zeta, cos_zeta, uvw, pqr), commanded_rpm, normalized_rpm)
                self.t = 0
            else: self.t += 1
        else: self.t += 1
        done = self.terminal()
        obs = self.get_obs((xyz, sin_zeta, cos_zeta, uvw, pqr), commanded_rpm, normalized_rpm)
        self.set_prev_dists((xyz, sin_zeta, cos_zeta, uvw, pqr), commanded_rpm, normalized_rpm)
        return obs, reward, done, info
    
    def reset(self):
        self.obs = super(TrajectoryEnv3D, self).reset()
        self.flagged = False
        return self.obs