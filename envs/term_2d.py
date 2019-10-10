import numpy as np
import gym
from gym import spaces

import math
from math import sin, cos, radians, sqrt, acos, exp

import pygame
import pygame.gfxdraw
import pygame.image as image
import envs.traj_2d as tenv

class TrajectoryEnvTerm2D(tenv.TrajectoryEnv2D):
    def __init__(self):
        super(TrajectoryEnvTerm2D, self).__init__()
        self.temperature = 5
        self.amplitude = 100

    def term_reward(self, state):
        xy, sin_zeta, cos_zeta, uv, r = state
        u = self.curr_dist
        if self.goal_counter <= self.traj_len-2:
            v = sum([(x-g)**2 for x, g in zip(xy, self.goal_list_xy[self.goal_counter+1])])**0.5
        else:
            v = 0
        dist = exp(-self.temperature*(u**2+u*v))
        rew = self.amplitude*dist
        return rew

    def step(self, data):
        thrust, rotation = data[0], data[1]
        thrust = np.clip(thrust, -1., 1.)
        rotation = np.clip(rotation, -1., 1.)
        xy, zeta, uv, r = self.player.step(thrust, rotation)
        term = data[-1]
        sin_zeta, cos_zeta = sin(zeta), cos(zeta)
        self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), data[:-1])
        reward, info = self.reward((xy, sin_zeta, cos_zeta, uv, r), data[:-1])
        if term == 1:
            if not self.flagged:
                term_rew = self.term_reward((xy, sin_zeta, cos_zeta, uv, r))
                if self.goal_counter == self.traj_len-1:
                    self.flagged = True
                else:
                    self.goal_counter += 1
                    self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), data[:-1])
                    self.t = 0.
            else: self.t += self.dt
        else:
            term_rew = 0.
            self.t += self.dt
        done = self.terminal()
        self.obs = self.get_obs((xy, sin_zeta, cos_zeta, uv, r), data[:-1])
        self.set_prev_dists((xy, sin_zeta, cos_zeta, uv, r), data[:-1])
        info.update({"term_rew" : term_rew})
        return self.obs, reward, done, info
    
    def reset(self):
        self.obs = super(TrajectoryEnvTerm2D, self).reset()
        self.flagged = False
        return self.obs
    
    def terminal(self):
        if self.curr_dist > 5: return True
        elif self.t >= self.T-self.dt: return True
        elif self.flagged: return True
        else: return False
        