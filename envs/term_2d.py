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
        #super(TrajectoryEnvTerm2D, self).reset()

    def reward(self, state, action):
        xy, sin_zeta, cos_zeta, uv, r = state

        # agent gets a negative reward based on how far away it is from the desired goal state
        dist_rew = 1/self.curr_dist if self.curr_dist > 1e-3 else 1/1e-3
        dist_rew += 10*(self.prev_dist-self.curr_dist)
        #dist_rew = -5*self.curr_dist**2
        sin_att_rew = 0*(self.prev_att_sin-self.curr_att_sin)
        cos_att_rew = 0*(self.prev_att_cos-self.curr_att_cos)
        vel_rew = 0*(self.prev_vel-self.curr_vel)
        ang_rew = 0*(self.prev_ang-self.curr_ang)

        att_rew = sin_att_rew+cos_att_rew

        # agent gets a negative reward for excessive action inputs
        ctrl_rew = -0*sum([a**2 for a in action])

        # derivative rewards
        ctrl_dev_rew = -0*sum([(a-b)**2 for a,b in zip(action, self.prev_action)])
        dist_dev_rew = -0.1*sum([(x-y)**2 for x, y in zip(xy, self.prev_xy)])
        sin_att_dev_rew = -0*(sin_zeta-sin(self.prev_zeta))**2
        cos_att_dev_rew = -0*(cos_zeta-cos(self.prev_zeta))**2
        vel_dev_rew = -0.1*sum([(u-v)**2 for u,v in zip(uv, self.prev_uv)])
        ang_dev_rew = -0.1*(r-self.prev_r)**2

        ctrl_rew += ctrl_dev_rew+vel_dev_rew+ang_dev_rew+dist_dev_rew+sin_att_dev_rew+cos_att_dev_rew

        # time reward to incentivize using the full time period
        time_rew = 1

        # calculate total reward
        total_reward = dist_rew+att_rew+vel_rew+ang_rew+ctrl_rew+time_rew
        return total_reward, {"dist_rew": dist_rew,
                                "att_rew": att_rew,
                                "vel_rew": vel_rew,
                                "ang_rew": ang_rew,
                                "ctrl_rew": ctrl_rew,
                                "dist_dev" : dist_dev_rew,
                                "att_dev" : sin_att_dev_rew+cos_att_dev_rew,
                                "vel_dev" : vel_dev_rew,
                                "ang_dev" : ang_dev_rew,
                                "time_rew": time_rew}

    def term_reward(self, state, term):
        xy, sin_zeta, cos_zeta, uv, r = state
        lp = -pi * self.curr_dist ** 2
        p = exp(lp)
        rew = term * lp + (1 - term) * log(1 - p + 1e-16)
        #print("term: ", term)
        #print("t rew: ", rew)
        #print()
        return rew

    def step(self, data):
        thrust, rotation = data[0], data[1]
        thrust += 0.5
        xy, zeta, uv, r = self.player.step(thrust, rotation)
        self.t += self.dt
        term = data[-1]
        sin_zeta, cos_zeta = sin(zeta), cos(zeta)
        self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), data[:-1])
        reward, info = self.reward((xy, sin_zeta, cos_zeta, uv, r), data[:-1])
        term_rew = self.term_reward((xy, sin_zeta, cos_zeta, uv, r), term)
        #reward += term_rew
        if term == 1:
            if self.goal_counter == self.traj_len-1:
                self.flagged = True
                goal_switch = 0.
            else:
                self.goal_counter += 1
                self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), data[:-1])
                self.t = 0.
                goal_switch = 1.
        else:
            goal_switch = 0.
        done = self.terminal()
        self.obs = self.get_obs((xy, sin_zeta, cos_zeta, uv, r), data[:-1])
        self.set_prev_dists()
        info.update({"term_rew" : term_rew})
        info.update({"goal_switch" : goal_switch})
        return self.obs, reward, done, info
    """
    def reset(self):
        self.flagged = False
        self.t = 0
        self.goal_counter = 0
        xy, zeta, uv, r = self.player.reset()
        #angle = np.random.RandomState().uniform(low=-pi/2, high=pi/2)
        #self.player.angle = angle * 180 / pi
        sin_zeta, cos_zeta = sin(zeta), cos(zeta)
        self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), np.zeros((2,)))
        self.set_prev_dists()
        self.obs = self.get_obs((xy, sin_zeta, cos_zeta, uv, r), np.zeros((2,)))
        return self.obs
    """
