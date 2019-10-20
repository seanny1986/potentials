import gym
import gym_aero
import gym_aero.envs.trajectory_env as trajectory_env
import random
from math import sin, cos, exp
from gym import spaces
import numpy as np

class TrajectoryEnvTerm(trajectory_env.TrajectoryEnv):
    def __init__(self):
        super(TrajectoryEnvTerm, self).__init__()
        self.num_fut_wp = 2
        state_size = (1+self.num_fut_wp)*15+5
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(state_size,))
        self.temperature = 5
        self.amplitude = 100

    def term_reward(self, state):
        xyz, sin_zeta, cos_zeta, uvw, pqr = state
        u = self.curr_dist
        if self.goal_counter <= self.traj_len-2:
            v = sum([(x-g)**2 for x, g in zip(xyz, self.goal_list_xyz[self.goal_counter+1])])**0.5
        else:
            v = 0
        dist = exp(-self.temperature*(u**2+u*v))
        rew = self.amplitude*dist
        return rew

    def step(self, data):
        action = data[:-1]
        term = data[-1]
        commanded_rpm = self.translate_action(action)
        xyz, zeta, xyz_dot, pqr = super(trajectory_env.TrajectoryEnv, self).step(commanded_rpm)        
        sin_zeta = [sin(z) for z in zeta]
        cos_zeta = [cos(z) for z in zeta]
        current_rpm = self.get_rpm()
        normalized_rpm = [rpm/self.max_rpm for rpm in current_rpm]
        self.set_curr_dists((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), commanded_rpm, normalized_rpm)
        reward, info = self.reward((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), commanded_rpm, normalized_rpm)
        term_rew = self.term_reward((xyz, sin_zeta, cos_zeta, xyz_dot, pqr)) if term == 1 else 0.
        if term == 1:
            if not self.flagged:
                term_rew = self.term_reward((xyz, sin_zeta, cos_zeta, xyz_dot, pqr))
                if self.goal_counter == self.traj_len-1:
                    self.flagged = True
                else:
                    self.goal_counter += 1
                    self.set_curr_dists((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), commanded_rpm, normalized_rpm)
                    self.t = 0.
            else:
                self.t += 1
                term_rew = 0.
        else:
            term_rew = 0.
            self.t += 1
        done = self.terminal()
        obs = self.get_obs((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), commanded_rpm, normalized_rpm)
        self.set_prev_dists((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), commanded_rpm, normalized_rpm)
        info.update({"term_rew" : term_rew})
        return obs, reward, done, info
    
    def reset(self):
        self.obs = super(TrajectoryEnvTerm, self).reset()
        self.flagged = False
        return self.obs