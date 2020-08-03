import numpy as np
import math
from math import sin, cos, tan, radians, degrees, acos, sqrt, pi, exp, copysign, atan2
from pygame.math import Vector2
import envs.traj_2d as traj_2d
import gym

import pygame
import pygame.gfxdraw

class FanTrajectoryEnv2D(traj_2d.TrajectoryEnv2D):
    def __init__(self):
        super(FanTrajectoryEnv2D, self).__init__()
        self.traj_len = 2
        self.num_fut_wp = 1
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        state_size = 18 + 2 * (self.num_fut_wp + 1)  # calculate the size of the state given the number of future waypoints in obs space
        self.observation_space = gym.spaces.Box(-1, 1, shape=(state_size,))
        self._max_episode_steps = 120
    
    def reward(self, state, action):
        xy, sin_zeta, cos_zeta, uv, r = state

        # agent gets a negative reward based on how far away it is from the desired goal state
        #dist_rew = 1/self.curr_dist**2 if self.curr_dist > self.goal_thresh else 1/self.goal_thresh**2
        delta1 = self.prev_dist-self.curr_dist
        delta2 = self.prev_dist_n-self.curr_dist_n
        dist_rew = 100 * (delta1 - 0.5 * delta2)
        sin_att_rew = 0*(self.prev_att_sin-self.curr_att_sin)
        cos_att_rew = 0*(self.prev_att_cos-self.curr_att_cos)
        vel_rew = 0*(self.prev_vel-self.curr_vel)
        ang_rew = 0*(self.prev_ang-self.curr_ang)

        att_rew = sin_att_rew+cos_att_rew

        # agent gets a negative reward for excessive action inputs
        ctrl_rew = -1*sum([a**2 for a in action])

        # derivative rewards
        ctrl_dev_rew = -1*sum([(a-b)**2 for a,b in zip(action, self.prev_action)])
        dist_dev_rew = -1*sum([(x-y)**2 for x,y in zip(xy, self.prev_xy)])
        sin_att_dev_rew = -0*(sin_zeta-sin(self.prev_zeta))**2
        cos_att_dev_rew = -0*(cos_zeta-cos(self.prev_zeta))**2
        vel_dev_rew = -1*sum([(u-v)**2 for u,v in zip(uv, self.prev_uv)])
        ang_dev_rew = -1*(r[0]-self.prev_r)**2

        ctrl_rew += ctrl_dev_rew+vel_dev_rew+ang_dev_rew+dist_dev_rew+sin_att_dev_rew+cos_att_dev_rew

        # time reward to incentivize using the full time period
        time_rew = -1

        # calculate total reward
        total_reward = dist_rew+att_rew+vel_rew+ang_rew+ctrl_rew+time_rew
        return total_reward, {"dist_rew": dist_rew,
                                "att_rew": att_rew,
                                "vel_rew": vel_rew,
                                "ang_rew": ang_rew,
                                "ctrl_rew": ctrl_rew,
                                "ctrl_dev_rew": ctrl_dev_rew,
                                "dist_dev" : dist_dev_rew,
                                "att_dev" : sin_att_dev_rew+cos_att_dev_rew,
                                "vel_dev" : vel_dev_rew,
                                "ang_dev" : ang_dev_rew,
                                "time_rew": time_rew}
    

    def set_curr_dists(self, state, action):
        xy, sin_zeta, cos_zeta, uv, r = state
        if not self.goal_counter > self.traj_len-1:
            self.curr_dist = sum([(x - g)**2 for x, g in zip(xy, self.goal_list_xy[self.goal_counter])])**0.5
            self.curr_dist_n = 0.
            if self.goal_counter < self.traj_len - 1:
                self.curr_dist_n = sum([(x - g)**2 for x, g in zip(xy, self.goal_list_xy[self.goal_counter + 1])])**0.5
        self.curr_att_sin = abs(sin_zeta-0.)
        self.curr_att_cos = abs(cos_zeta-1.)
        self.curr_vel = sum([(u - 0.)**2 for u in uv])**0.5
        self.curr_ang = abs(r[0] - 0.)
        self.curr_action = action
        self.curr_xy = xy
        self.curr_zeta = self.player.angle
        self.curr_uv = uv
        self.curr_r = r[0]
    
    def set_prev_dists(self):
        self.prev_dist = self.curr_dist
        self.prev_dist_n = self.curr_dist_n
        self.prev_att_sin = self.curr_att_sin
        self.prev_att_cos = self.curr_att_cos
        self.prev_vel = self.curr_vel
        self.prev_ang = self.curr_ang
        self.prev_action = self.curr_action

        self.prev_xy = self.curr_xy
        self.prev_zeta = self.curr_zeta
        self.prev_uv = self.curr_uv
        self.prev_r = self.curr_r

    def get_obs(self, state, action):
        xy, sin_zeta, cos_zeta, uv, r = state
        angle = self.player.angle
        xy_obs = []
        sin_zeta_obs = []
        cos_zeta_obs = []
        uv_obs = []
        r_obs = []
        for i in range(self.num_fut_wp+1):
            if self.goal_counter + i <= len(self.goal_list_xy) - 1:
                xy_obs = xy_obs + self.rotate([g - x for x, g in zip(xy, self.goal_list_xy[self.goal_counter + i])], angle)
            else:
                xy_obs = xy_obs + [0., 0.]

        sin_zeta_obs = [0. - sin_zeta]
        cos_zeta_obs = [1. - cos_zeta]
        uv_obs = [0. - u for u in uv]
        r_obs = [0. - r[0]]
        
        # derivatives
        dv_dt = [(u - v) / self.dt for u, v in zip(uv, self.prev_uv)]
        dr_dt = [(r[0] - self.prev_r) / self.dt]
        derivatives = uv + dv_dt + dr_dt

        # actions
        da_dt = [(a-b)/self.dt for a, b in zip(action, self.prev_action)]
        tar_obs = xy_obs + sin_zeta_obs + cos_zeta_obs + uv_obs + r_obs + derivatives
        thrust = [self.player.thrust]
        steering = [self.player.steering]
        steering_angle = [self.player.steering_angle]
        next_state = tar_obs + action + da_dt + thrust + steering + steering_angle + [self.t]
        return next_state

    def term_reward(self, term):
        term_rew = 0 if self.curr_dist > self.goal_thresh else 10
        return term_rew
    
    def terminal(self):
        #if self.curr_dist > 5: return True
        if (self.goal_counter == self.traj_len - 1) and (self.curr_dist) < self.goal_thresh: return True
        if self.t >= self.T-self.dt: return True
        else: return False
        
    def reset(self):
        self.flagged = False
        self.t = 0
        self.goal_counter = 0
        self.goal_list_xy = []
        xy_ = np.array([1.5, 0.])
        self.goal_list_xy.append(list(xy_.copy()))
        for _ in range(self.traj_len-1):
            angle = np.random.RandomState().uniform(low=-pi, high=-pi/2)
            flip = np.random.RandomState().uniform(low=0, high=1)
            a = 1 if flip >= 0.5 else -1
            angle *= a
            temp = np.array([1.5*cos(angle), 1.5*sin(angle)])
            xy_ += temp
            self.goal_list_xy.append(list(xy_.copy()))
        
        xy, zeta, uv, r = self.player.reset()
        angle = np.random.RandomState().uniform(low=-pi/4, high=pi/4)
        #angle = zeta[0]
        self.player.angle = angle
        sin_zeta, cos_zeta = sin(angle), cos(angle)
        self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), [0., 0.])
        self.set_prev_dists()
        self.obs = self.get_obs((xy, sin_zeta, cos_zeta, uv, r), [0., 0.])
        return self.obs
    
    def render(self, close=False):

        def arrow(screen, lcolor, tricolor, start, end, trirad):
            pygame.draw.line(screen, lcolor, start, end, 2) # draw the line
            rotation = atan2(start[1] - end[1], end[0] - start[0]) + pi/2 # calculate rotation angle
            pygame.draw.polygon(screen, 
                                tricolor, 
                                ((end[0] + trirad * sin(rotation), end[1] + trirad * cos(rotation)),                                     
                                 (end[0] + trirad * sin(rotation-+2*pi/3), end[1] + trirad * cos(rotation-2*pi/3)), 
                                 (end[0] + trirad * sin(rotation+2*pi/3), end[1] + trirad * cos(rotation+2*pi/3))))

        if not self.init:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_dim[0], self.window_dim[1]))
            pygame.display.set_caption('Trajectory 2D')
            self.font = pygame.font.Font('freesansbold.ttf', 14)
            self.init = True
        else:
            for event in pygame.event.get():
                pass
            self.screen.fill([255, 255, 255])

            curr_goals = self.obs[:6]
            start = [self.curr_xy[0]*self.scaling + self.WINDOW_SIZE/3, self.curr_xy[1]*self.scaling+self.WINDOW_SIZE/2]
            for i in range(self.num_fut_wp + 1):
                vec = curr_goals[2*i:2*i+2]
                mag = sum([x**2 for x in vec])**0.5
                if not mag == 0:
                    normed = [0.5 * x for x in vec]
                    uv = self.rotate(normed, -self.curr_zeta)
                    end = [u*self.scaling + x for u, x in zip(uv, start)]
                    arrow(self.screen, (200, 0, 50), (200, 0, 50), start, end, 0.1 * self.scaling)

            thrust_mag = 2 * self.player.thrust / self.player.max_thrust
            steering_angle = self.player.steering_angle
            beta = self.curr_zeta + steering_angle
            vec = self.rotate([thrust_mag, 0], -beta)
            end = [u*self.scaling + x for u, x in zip(vec, start)]
            arrow(self.screen, (180, 180, 180), (180, 180, 180), start, end, 0.1 * self.scaling)

            prev_x = int(0+self.WINDOW_SIZE/3)
            prev_y = int(0+self.WINDOW_SIZE/2)
            pygame.gfxdraw.filled_circle(self.screen, 
                                        prev_x, 
                                        prev_y, 
                                        int(0.1*self.scaling), 
                                        (0, 255, 0))
            for i, g in enumerate(self.goal_list_xy):
                curr_x = int(self.scaling*g[0]+self.WINDOW_SIZE/3)
                curr_y = int(self.scaling*g[1]+self.WINDOW_SIZE/2)
                if i == self.goal_counter:
                    pygame.draw.circle(self.screen,
                                        (0, 0, 0),
                                        (curr_x, curr_y),
                                        int(0.46*self.scaling),
                                        1)
                    colour = (255,0,0)
                else: colour = (0, 255, 0)
                pygame.draw.line(self.screen,
                                colour,
                                (prev_x, prev_y),
                                (curr_x, curr_y),
                                3)
                pygame.gfxdraw.filled_circle(self.screen, 
                                        curr_x, 
                                        curr_y, 
                                        int(0.1*self.scaling), 
                                        colour)
                prev_x = curr_x
                prev_y = curr_y
            
            pts = self.player.update_gfx()
            player_pos = [[self.scaling*p[0]+self.WINDOW_SIZE/3, self.scaling*p[1]+self.WINDOW_SIZE/2] for p in pts]
            pygame.gfxdraw.filled_trigon(self.screen, 
                                        int(player_pos[0][0]), int(player_pos[0][1]), 
                                        int(player_pos[1][0]), int(player_pos[1][1]),
                                        int(player_pos[2][0]), int(player_pos[2][1]),
                                        self.player.colour)
            
            # dump data to screen for debugging
            
            time_text = self.font.render("Timestamp: {:.2f}".format(self.t), False, (0,0,0))
            frame_text = self.font.render("Frame: {}".format(int(self.t/self.dt)), False, (0,0,0))
            goal_text = self.font.render("Goal index: {}".format(self.goal_counter), False, (0,0,0))
            curr_goal_text = self.font.render("Current Goal Position: [{:.2f}, {:.2f}]".format(self.goal_list_xy[self.goal_counter][0], self.goal_list_xy[self.goal_counter][1]), False, (0,0,0))
            space_text = self.font.render(" ", False, (0,0,0))
            state_text = self.font.render("-------- State --------", False, (0,0,0))
            action_text = self.font.render("Thrust: [{:.2f}], Steering: [{:.2f}]".format(self.curr_action[0], self.curr_action[1]), False, (0,0,0))
            drag_text = self.font.render("Drag: [{:.2f}]".format(-self.player.drag), False, (0,0,0))
            pos_text = self.font.render("Position: [{:.2f}, {:.2f}]".format(self.player.position.x, self.player.position.y), False, (0,0,0))
            uv_text = self.font.render("Linear velocity: [{:.2f}, {:.2f}]".format(self.player.velocity.x,self.player.velocity.y), False, (0,0,0))
            accel_text = self.font.render("Linear acceleration: [{:.2f}, {:.2f}]".format(self.player.acceleration, 0.), False, (0,0,0))
            angle_text = self.font.render("Zeta: [{:.2f}]".format(degrees(self.player.angle)), False, (0,0,0))
            r_text = self.font.render("Angular velocity: [{:.2f}]".format(self.player.angular_velocity), False, (0,0,0))
            transition_probability_text = self.font.render("Transition PDF: {:.8f}".format(self.pdf_norm*exp(-self.temperature*self.curr_dist**2)), False, (0,0,0))
            
            break_text = self.font.render("-------- Observation --------", False, (0,0,0))
            obs_pos_text = self.font.render("Position Vector: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[0], self.obs[1], self.obs[2], self.obs[3]), False, (0,0,0))
            obs_zeta_text = self.font.render("Zeta Vector: [{:.2f}, {:.2f}]".format(self.obs[4], self.obs[5]), False, (0,0,0))
            obs_vel_text = self.font.render("Velocity Vector: [{:.2f}, {:.2f}]".format(self.obs[6], self.obs[7]), False, (0,0,0))
            obs_omega_text = self.font.render("Omega Vector: [{:.2f}]".format(self.obs[8]), False, (0,0,0))
            obs_derivative_uv_text = self.font.render("Derivatives -- UV: [{:.2f}, {:.2f}]".format(self.obs[9], self.obs[10]), False, (0,0,0))
            obs_derivative_duv_text = self.font.render("Derivatives -- dUV_dt: [{:.2f}, {:.2f}]".format(self.obs[11], self.obs[12]), False, (0,0,0))
            obs_derivative_domega_text = self.font.render("Derivatives -- dR_dt: [{:.2f}]".format(self.obs[13]), False, (0,0,0))
            obs_action_text = self.font.render("Action: [{:.2f}, {:.2f}]".format(self.obs[14], self.obs[15]), False, (0,0,0))
            obs_action_derivative_text = self.font.render("Action Derivative: [{:.2f}, {:.2f}]".format(self.obs[16], self.obs[17]), False, (0,0,0))
            obs_thrust_text = self.font.render("Thrust: [{:.2f}]".format(self.obs[18]), False, (0,0,0))
            obs_steering_text = self.font.render("Steering: [{:.2f}]".format(self.obs[19]), False, (0,0,0))
            obs_steering_angle_text = self.font.render("Steering Angle: [{:.2f}]".format(self.obs[20]), False, (0,0,0))
            obs_time_text = self.font.render("Time: [{:.2f}]".format(self.obs[21]), False, (0,0,0))
            
            text_list = [time_text, frame_text, goal_text, curr_goal_text,
                         space_text,
                         state_text,
                         action_text, drag_text, pos_text, transition_probability_text, uv_text, accel_text, angle_text, r_text, 
                         space_text,
                         break_text,
                         obs_pos_text, obs_zeta_text, obs_vel_text, obs_omega_text,
                         obs_derivative_uv_text, obs_derivative_duv_text, obs_derivative_domega_text,
                         obs_action_text, obs_action_derivative_text, obs_thrust_text, obs_steering_text, obs_steering_angle_text, obs_time_text]
            for i, t in enumerate(text_list):
                text_rect = t.get_rect()
                text_rect.left += 5
                text_rect.top += 5 + text_rect.height + i * 25
                self.screen.blit(t, text_rect)
            
            pygame.display.flip()