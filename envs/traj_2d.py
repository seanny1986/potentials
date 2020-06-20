import numpy as np
import gym
from gym import spaces

import math
from math import sin, cos, tan, radians, degrees, acos, sqrt, pi, exp, copysign, atan2
from pygame.math import Vector2

import pygame
import pygame.gfxdraw

class TrajectoryEnv2D(gym.Env):
    def __init__(self, dt=0.05):
        self.dt = dt                                                    # simulation timestep
        self.player = Player(dt=dt)                                          # list of player characters in the environment
        self.DIM = [15, 15]
        self.temperature = 15
        self.T = 3
        self.goal_thresh = 1e-1
        self.traj_len = 5

        self.num_fut_wp = 2
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        state_size = 10+7*(self.num_fut_wp+1)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(state_size,))
        
        self.WINDOW_SIZE = 1000
        self.WINDOW_RANGE = self.DIM[0]
        self.window_dim = np.array([self.WINDOW_SIZE, self.WINDOW_SIZE])
        self.scaling = self.WINDOW_SIZE/self.WINDOW_RANGE
        self.target_size = int(self.scaling*0.1)

        self.init = False

        print("Trajectory Env 2D initialized.")
    
    def rotate(self, vec, angle):
        """
        Rotates 2D vector by angle theta
        """
        u, v = vec
        cz = cos(angle)
        sz = sin(angle)
        x = u * cz - v * sz
        y = u * sz + v * cz
        return [x, y]
    
    def get_goal_positions(self):
        n = int(2*(self.num_fut_wp+1))
        return self.obs[:n]

    def switch_goal(self, state):
        if self.curr_dist < self.goal_thresh: return True
        else: return False

    def reward(self, state, action):
        xy, sin_zeta, cos_zeta, uv, r = state

        # agent gets a negative reward based on how far away it is from the desired goal state
        #dist_rew = 1/self.curr_dist if self.curr_dist > self.goal_thresh else 100/self.goal_thresh
        dist_rew = 100*(self.prev_dist-self.curr_dist)
        sin_att_rew = 0*(self.prev_att_sin-self.curr_att_sin)
        cos_att_rew = 0*(self.prev_att_cos-self.curr_att_cos)
        vel_rew = 10*(self.prev_vel-self.curr_vel)
        ang_rew = 1*(self.prev_ang-self.curr_ang)

        att_rew = sin_att_rew+cos_att_rew

        # agent gets a negative reward for excessive action inputs
        ctrl_rew = -1e-6*sum([a**2 for a in action])

        # derivative rewards
        ctrl_dev_rew = -1e-5*sum([(a-b)**2 for a,b in zip(action, self.prev_action)])
        dist_dev_rew = -1e-2*sum([(x-y)**2 for x, y in zip(xy, self.prev_xy)])
        sin_att_dev_rew = -0*(sin_zeta-sin(self.prev_zeta))**2
        cos_att_dev_rew = -0*(cos_zeta-cos(self.prev_zeta))**2
        vel_dev_rew = -0*sum([(u-v)**2 for u,v in zip(uv, self.prev_uv)])
        ang_dev_rew = -0*(r[0]-self.prev_r)**2

        ctrl_rew += ctrl_dev_rew+vel_dev_rew+ang_dev_rew+dist_dev_rew+sin_att_dev_rew+cos_att_dev_rew

        # time reward to incentivize using the full time period
        time_rew = 0

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
    
    def term_reward(self, state):
        xy, sin_zeta, cos_zeta, uv, r = state
        if self.curr_dist < self.goal_thresh and not self.flagged:
            rew = 100
        else: return 0.
        return rew

    def terminal(self):
        if self.curr_dist > 5: return True
        if self.t >= self.T-self.dt: return True
        else: return False
    
    def set_curr_dists(self, state, action):
        xy, sin_zeta, cos_zeta, uv, r = state
        if not self.goal_counter > self.traj_len-1:
            self.curr_dist = sum([(x-g)**2 for x, g in zip(xy, self.goal_list_xy[self.goal_counter])])**0.5
            self.curr_att_sin = abs(sin_zeta-sin(self.goal_list_zeta[self.goal_counter]))
            self.curr_att_cos = abs(cos_zeta-cos(self.goal_list_zeta[self.goal_counter]))
            self.curr_vel = sum([(u-g)**2 for u, g in zip(uv, self.goal_list_uv[self.goal_counter])])**0.5
            self.curr_ang = abs(r[0]-self.goal_list_r[self.goal_counter])
        self.curr_action = action
        self.curr_xy = xy
        self.curr_zeta = acos(cos_zeta)
        self.curr_uv = uv
        self.curr_r = r[0]
        #print(self.curr_dist)
        
    def set_prev_dists(self):
        self.prev_dist = self.curr_dist
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
        angle = acos(cos_zeta)
        xy_obs = []
        sin_zeta_obs = []
        cos_zeta_obs = []
        uv_obs = []
        r_obs = []
        for i in range(self.num_fut_wp+1):
            if self.goal_counter+i <= len(self.goal_list_xy)-1:
                xy_obs = xy_obs+self.rotate([x-g for x, g in zip(xy, self.goal_list_xy[self.goal_counter+i])], angle)
                sin_zeta_obs = sin_zeta_obs+[sin_zeta-sin(self.goal_list_zeta[self.goal_counter+i])]
                cos_zeta_obs = cos_zeta_obs+[cos_zeta-cos(self.goal_list_zeta[self.goal_counter+i])]
                uv_obs = uv_obs+[u-g for u, g in zip(uv, self.goal_list_uv[self.goal_counter+i])]
                r_obs = r_obs+[r[0]-self.goal_list_r[self.goal_counter+i]]
            else:
                xy_obs = xy_obs+[0., 0.]
                sin_zeta_obs = sin_zeta_obs+[0.]
                cos_zeta_obs = cos_zeta_obs+[0.]
                uv_obs = uv_obs+[0., 0.]
                r_obs = r_obs+[0.]

        # derivatives
        dv_dt = [(u-v)/self.dt for u, v in zip(uv, self.prev_uv)]
        dr_dt = [(r[0]-self.prev_r)/self.dt]
        derivatives = uv+dv_dt+dr_dt

        # actions
        da_dt = [(a-b)/self.dt for a, b in zip(action, self.prev_action)]
        tar_obs = xy_obs+sin_zeta_obs+cos_zeta_obs+uv_obs+r_obs+derivatives
        next_state = tar_obs+action+da_dt+[self.t]
        return next_state
    
    def translate_action(self, actions):
        # update thrust and clip to max thrust percentage
        thrust_c, steering_c = actions
        thrust_c += 0.25 * self.player.max_thrust
        steering_c *= 100
        return thrust_c, steering_c

    def step(self, data):
        actions = list(data)
        thrust, steering = self.translate_action(actions)
        xy, zeta, uv, r = self.player.step(thrust, steering)
        sin_zeta, cos_zeta = sin(zeta[0]), cos(zeta[0])
        self.t += self.dt
        self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), [thrust, steering])
        reward, info = self.reward((xy, sin_zeta, cos_zeta, uv, r), [thrust, steering])
        term = self.switch_goal((xy, sin_zeta, cos_zeta, uv, r))
        if term:
            term_rew = self.term_reward((xy, sin_zeta, cos_zeta, uv, r))
            if self.goal_counter < self.traj_len-1:
                self.goal_counter += 1
                self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), [thrust, steering])
                self.t = 0.
        else:
            term_rew = 0.
        reward += term_rew
        done = self.terminal()
        self.obs = self.get_obs((xy, sin_zeta, cos_zeta, uv, r), [thrust, steering])
        self.set_prev_dists()
        return self.obs, reward, done, info
    
    def reset(self):
        self.flagged = False
        self.t = 0
        self.goal_counter = 0
        self.goal_list_xy = []
        angle = np.random.RandomState().uniform(low=-pi/3, high=pi/3)
        rad = np.random.RandomState().uniform(1, 2.5)
        xy_ = np.array([rad*cos(angle), rad*sin(angle)])
        self.goal_list_xy.append(xy_.copy())
        for _ in range(self.traj_len-1):
            angle = np.random.RandomState().uniform(low=-pi/3, high=pi/3)
            rad = np.random.RandomState().uniform(1, 2.5)
            temp = np.array([rad*cos(angle), rad*sin(angle)])
            xy_ += temp
            self.goal_list_xy.append(list(xy_.copy()))

        self.goal_list_zeta = []
        self.goal_list_uv = []
        self.goal_list_r = []
        for i in range(self.traj_len):
            self.goal_list_zeta.append(0.)
            self.goal_list_uv.append(list(np.zeros((2,))))
            self.goal_list_r.append(0.)
        
        xy, zeta, uv, r = self.player.reset()
        angle = np.random.RandomState().uniform(low=-pi/3, high=pi/3)
        self.player.angle = angle
        sin_zeta, cos_zeta = sin(angle), cos(angle)
        self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), [0., 0.])
        self.set_prev_dists()
        self.obs = self.get_obs((xy, sin_zeta, cos_zeta, uv, r), [0., 0.])
        return self.obs
    
    def render(self, close=False):

        def arrow(screen, lcolor, tricolor, start, end, trirad):
            pygame.draw.line(screen, lcolor, start, end, 2) # draw the line
            rotation = degrees(atan2(start[1] - end[1], end[0] - start[0])) + 90 # calculate rotation angle
            pygame.draw.polygon(screen, 
                                tricolor, 
                                (
                                    (
                                        end[0] + trirad * sin(radians(rotation)), 
                                        end[1] + trirad * cos(radians(rotation))
                                    ), 
                                    
                                    (
                                        end[0] + trirad * sin(radians(rotation - 120)), 
                                        end[1] + trirad * cos(radians(rotation - 120))
                                    ), 
                                    
                                    (
                                        end[0] + trirad * sin(radians(rotation + 120)), 
                                        end[1] + trirad * cos(radians(rotation + 120))
                                    )
                                )
                                )

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
            angle_rad = -self.curr_zeta
            for i in range(self.num_fut_wp + 1):
                vec = curr_goals[2*i:2*i+2]
                mag = sum([x**2 for x in vec])**0.5
                if not mag == 0:
                    normed = [-x/mag for x in vec]
                    uv = self.rotate(normed, angle_rad)
                    start = [self.curr_xy[0]*self.scaling + self.WINDOW_SIZE/3, self.curr_xy[1]*self.scaling+self.WINDOW_SIZE/2]
                    end = [u*self.scaling + x for u, x in zip(uv, start)]
                    arrow(self.screen, (200, 0, 50), (200, 0, 50), start, end, 0.1 * self.scaling)

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
            pos_text = self.font.render("Position: [{:.2f}, {:.2f}]".format(self.player.position.x, self.player.position.y), False, (0,0,0))
            uv_text = self.font.render("Linear velocity: [{:.2f}, {:.2f}]".format(self.player.velocity.x,self.player.velocity.y), False, (0,0,0))
            angle_text = self.font.render("Zeta: [{:.2f}]".format(radians(self.player.angle)), False, (0,0,0))
            r_text = self.font.render("Rotational velocity: [{:.2f}]".format(self.player.angular_velocity), False, (0,0,0))
            action_text = self.font.render("Thrust: [{:.2f}], Steering: [{:.2f}]".format(self.curr_action[0], self.curr_action[1]), False, (0,0,0))
            
            break_text = self.font.render("-------- Observation --------", False, (0,0,0))
            obs_pos_text = self.font.render("Position Vector: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[0], self.obs[1], self.obs[2], self.obs[3], self.obs[4], self.obs[5]), False, (0,0,0))
            obs_zeta_text = self.font.render("Zeta Vector: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[6], self.obs[7], self.obs[8], self.obs[9], self.obs[10], self.obs[11]), False, (0,0,0))
            obs_vel_text = self.font.render("Velocity Vector: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[12], self.obs[13], self.obs[14], self.obs[15], self.obs[16], self.obs[17]), False, (0,0,0))
            obs_omega_text = self.font.render("Omega Vector: [{:.2f}, {:.2f}, {:.2f}]".format(self.obs[18], self.obs[19], self.obs[20]), False, (0,0,0))
            obs_derivative_uv_text = self.font.render("Derivatives -- UV: [{:.2f}, {:.2f}]".format(self.obs[21], self.obs[22]), False, (0,0,0))
            obs_derivative_duv_text = self.font.render("Derivatives -- dUV_dt: [{:.2f}, {:.2f}]".format(self.obs[23], self.obs[24]), False, (0,0,0))
            obs_derivative_domega_text = self.font.render("Derivatives -- dR_dt: [{:.2f}]".format(self.obs[25]), False, (0,0,0))
            obs_action_text = self.font.render("Action: [{:.2f}, {:.2f}]".format(self.obs[26], self.obs[27]), False, (0,0,0))
            obs_action_derivative_text = self.font.render("Action Derivative: [{:.2f}, {:.2f}]".format(self.obs[28], self.obs[29]), False, (0,0,0))
            obs_time_text = self.font.render("Time: [{:.2f}]".format(self.obs[30]), False, (0,0,0))
            
            text_list = [time_text, frame_text, goal_text, curr_goal_text,
                         space_text,
                         state_text,
                         pos_text, uv_text, angle_text, r_text, action_text,
                         space_text,
                         break_text,
                         obs_pos_text, obs_zeta_text, obs_vel_text, obs_omega_text,
                         obs_derivative_uv_text, obs_derivative_duv_text, obs_derivative_domega_text,
                         obs_action_text, obs_action_derivative_text, obs_time_text]
            for i, t in enumerate(text_list):
                text_rect = t.get_rect()
                text_rect.left += 5
                text_rect.top += 5 + text_rect.height + i * 25
                self.screen.blit(t, text_rect)

            pygame.display.flip()


class Player:
    def __init__(self, x=0., y=0., angle=0., length=0.3, max_steering_angle=30., max_thrust=7.5, dt=0.05):
        self.length = length
        self.max_thrust = max_thrust
        self.max_steering_angle = max_steering_angle
        self.max_acceleration = 5.
        self.max_velocity = 3
        self.tau_thrust = 0.5
        self.tau_steering = 0.5
        self.mass = 1.
        self.cd = 1.
        self.dt = dt

        # simulation parameters
        self.position = Vector2(x, y)
        self.velocity = Vector2(0., 0.)
        self.angle = angle
        self.angular_velocity = 0.
        self.acceleration = 0.
        self.steering_angle = 0.
        self.thrust = 0.

        # points for rendering
        self.pts = [[0.2, 0.], [-0.1, 0.1], [-0.1, -0.1]]
        self.colour = (0, 0, 255)

    def step(self, thrust_c, steering_c):
        # update thrust and clip to max steering percentage
        self.thrust = self.tau_thrust * thrust_c + (1 - self.tau_thrust) * self.thrust
        self.thrust = np.clip(self.thrust, 0., self.max_thrust)

        # update steering and clip to max steering percentage
        self.steering_angle = self.tau_steering * steering_c + (1 - self.tau_steering) * self.steering_angle
        self.steering_angle = np.clip(self.steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # update acceleration and steering angle
        self.acceleration = (self.thrust - self.cd * self.velocity.x ** 2) / self.mass
        #self.acceleration = np.clip(self.acceleration, -self.max_acceleration, self.max_acceleration)
        
        # update linear velocity and clamp
        self.velocity += (self.acceleration * self.dt, 0)
        if self.velocity.x < 0: self.velocity.x = 0.
        if self.velocity.x > self.max_velocity: self.velocity.x = self.max_velocity
        
        # update angular velocity
        if not self.steering_angle == 0:
            turning_radius = self.length / tan(radians(self.steering_angle))
            #print(turning_radius)
            self.angular_velocity = self.velocity.x / turning_radius
        else:
            self.angular_velocity = 0.

        # update position and angle
        self.position += self.velocity.rotate(-self.angle) * self.dt
        self.angle += degrees(self.angular_velocity) * self.dt
        
#        print("thrust command: ", thrust_c)
#        print("thrust: ", self.thrust)
#        print("drag: ", self.cd * self.velocity.x ** 2)
#        print("acceleration: ", self.acceleration)
#        print("steering command: ", steering_c)
#        print("steering angle: ", self.steering_angle)
#        print()
#        input()

        # get new values and return
        position = [self.position.x, self.position.y]
        angle = [radians(self.angle)]
        velocity = [self.velocity.x, self.velocity.y]
        angular_velocity = [self.angular_velocity] 
        return position, angle, velocity, angular_velocity 
    
    def update_gfx(self):
        rad_angle = -radians(self.angle)
        pts = []
        for p in self.pts:
            x_ = self.position.x +(p[0] * cos(rad_angle) - p[1] * sin(rad_angle))
            y_ = self.position.y + (p[0] * sin(rad_angle) + p[1] * cos(rad_angle))
            pts.append([x_, y_])
        return pts

    def reset(self):
        self.position = Vector2(0., 0.)
        self.velocity = Vector2(0., 0.)
        self.angle = 0.
        self.angular_velocity = 0.
        self.acceleration = 0.
        self.steering_angle = 0.
        self.thrust = 0.
        position = [self.position.x, self.position.y]
        angle = [radians(self.angle)]
        velocity = [self.velocity.x, self.velocity.y]
        angular_velocity = [self.angular_velocity] 
        return position, angle, velocity, angular_velocity
