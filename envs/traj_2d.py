import numpy as np
import gym
from gym import spaces

import math
from math import sin, cos, tan, radians, degrees, acos, sqrt, pi, exp, copysign
from pygame.math import Vector2

import pygame
import pygame.gfxdraw

class TrajectoryEnv2D(gym.Env):
    def __init__(self, dt=0.05):
        self.dt = dt                                                    # simulation timestep
        self.player = Player(dt=dt)                                          # list of player characters in the environment
        self.DIM = np.array([10, 10])
        self.temperature = 10
        self.T = 3
        self.goal_thresh = 1e-1
        self.traj_len = 4

        self.num_fut_wp = 2
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        state_size = 10+7*(self.num_fut_wp+1)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(state_size,))
        
        self.WINDOW_SIZE = 800
        self.WINDOW_RANGE = self.DIM[0]
        self.window_dim = np.array([self.WINDOW_SIZE, self.WINDOW_SIZE])
        self.scaling = self.WINDOW_SIZE/self.WINDOW_RANGE
        self.target_size = int(self.scaling*0.1)

        self.init = False

        print("Trajectory Env 2D initialized.")
    
    def rotate(self, vec, angle):
        cz = cos(angle)
        sz = sin(angle)
        x = vec[0]*cz-vec[1]*sz
        y = vec[0]*sz+vec[1]*cz
        return [x, y]

    def get_inertial_pos(self):
        return np.array([self.player.position.x, self.player.position.y])
    
    def get_goal_positions(self, n=2):
        goals = []
        for i in range(n):
            if self.goal_counter < self.traj_len-n:
                goals = goals+[self.goal_list_xy[self.goal_counter+i]]
            else: goals = goals+[np.zeros((2,))]
        return np.hstack(goals)

    def switch_goal(self, state):
        if self.curr_dist < self.goal_thresh: return True
        else: return False

    def reward(self, state, action):
        xy, sin_zeta, cos_zeta, uv, r = state

        # agent gets a negative reward based on how far away it is from the desired goal state
        #dist_rew = 1/self.curr_dist if self.curr_dist > self.goal_thresh else 1/self.goal_thresh
        dist_rew = 100*(self.prev_dist-self.curr_dist)
        #dist_rew = 10*exp(-self.curr_dist**2)
        sin_att_rew = 0*(self.prev_att_sin-self.curr_att_sin)
        cos_att_rew = 0*(self.prev_att_cos-self.curr_att_cos)
        vel_rew = 1*(self.prev_vel-self.curr_vel)
        ang_rew = 1*(self.prev_ang-self.curr_ang)

        att_rew = sin_att_rew+cos_att_rew

        # agent gets a negative reward for excessive action inputs
        ctrl_rew = -0*sum([a**2 for a in action])

        # derivative rewards
        ctrl_dev_rew = -0*sum([(a-b)**2 for a,b in zip(action, self.prev_action)])
        dist_dev_rew = -10*sum([(x-y)**2 for x, y in zip(xy, self.prev_xy)])
        sin_att_dev_rew = -0*(sin_zeta-sin(self.prev_zeta))**2
        cos_att_dev_rew = -0*(cos_zeta-cos(self.prev_zeta))**2
        vel_dev_rew = -10*sum([(u-v)**2 for u,v in zip(uv, self.prev_uv)])
        ang_dev_rew = -10*(r-self.prev_r)**2

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

    def terminal(self):
        if self.curr_dist > 5: return True
        elif self.t >= self.T-self.dt: return True
        else: return False
    
    def set_curr_dists(self, state, action):
        xy, sin_zeta, cos_zeta, uv, r = state
        if not self.goal_counter > self.traj_len-1:
            self.curr_dist = sum([(x-g)**2 for x, g in zip(xy, self.goal_list_xy[self.goal_counter])])**0.5
            self.curr_att_sin = abs(sin_zeta-sin(self.goal_list_zeta[self.goal_counter]))
            self.curr_att_cos = abs(cos_zeta-cos(self.goal_list_zeta[self.goal_counter]))
            self.curr_vel = sum([(u-g)**2 for u, g in zip(uv, self.goal_list_uv[self.goal_counter])])**0.5
            self.curr_ang = abs(r-self.goal_list_r[self.goal_counter])
        self.curr_action = action
        #print(self.curr_dist)
        
    def set_prev_dists(self, state, action):
        xy, _, cos_zeta, uv, r = state
        self.prev_dist = self.curr_dist
        self.prev_att_sin = self.curr_att_sin
        self.prev_att_cos = self.curr_att_cos
        self.prev_vel = self.curr_vel
        self.prev_ang = self.curr_ang
        self.prev_action = self.curr_action

        self.prev_xy = xy
        self.prev_zeta = acos(cos_zeta)
        self.prev_uv = uv
        self.prev_r = r
    
    def get_obs(self, state, action):
        xy, sin_zeta, cos_zeta, uv, r = state
        
        xy_obs = []
        sin_zeta_obs = []
        cos_zeta_obs = []
        uv_obs = []
        r_obs = []
        for i in range(self.num_fut_wp+1):
            if self.goal_counter+i <= len(self.goal_list_xy)-1:
                xy_obs = xy_obs+[x-g for x,g in zip(xy, self.goal_list_xy[self.goal_counter+i])]
                sin_zeta_obs = sin_zeta_obs+[sin_zeta-sin(self.goal_list_zeta[self.goal_counter+i])]
                cos_zeta_obs = cos_zeta_obs+[cos_zeta-cos(self.goal_list_zeta[self.goal_counter+i])]
                uv_obs = uv_obs+self.rotate([u-g for u, g in zip(uv, self.goal_list_uv[self.goal_counter+i])], acos(cos_zeta))
                r_obs = r_obs+[r-self.goal_list_r[self.goal_counter+i]]
            else:
                xy_obs = xy_obs+[0., 0.]
                sin_zeta_obs = sin_zeta_obs+[0.]
                cos_zeta_obs = cos_zeta_obs+[0.]
                uv_obs = uv_obs+[0., 0.]
                r_obs = r_obs+[0.]
        
        # derivatives
        dv_dt = [(u-v)/self.dt for u, v in zip(uv, self.prev_uv)]
        dr_dt = [(r-self.prev_r)/self.dt]
        derivatives = uv.tolist()+dv_dt+dr_dt

        # actions
        a = action.tolist()
        da_dt = ((action-self.prev_action)/self.dt).tolist()

        tar_obs = xy_obs+sin_zeta_obs+cos_zeta_obs+uv_obs+r_obs+derivatives
        next_state = tar_obs+a+da_dt+[self.t]
        return next_state

    def step(self, data):
        thrust, rotation = data[0], data[1]
        xy, zeta, uv, r = self.player.step(thrust, rotation)
        sin_zeta, cos_zeta = sin(zeta), cos(zeta)
        self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), data)
        reward, info = self.reward((xy, sin_zeta, cos_zeta, uv, r), data)
        term = self.switch_goal((xy, sin_zeta, cos_zeta, uv, r))
        if term:
            if self.goal_counter < self.traj_len-1:
                self.goal_counter += 1
                self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), data)
                self.t = 0
            else: self.t += 1
        else: self.t += self.dt
        done = self.terminal()
        self.obs = self.get_obs((xy, sin_zeta, cos_zeta, uv, r), data)
        self.set_prev_dists((xy, sin_zeta, cos_zeta, uv, r), data)
        return self.obs, reward, done, info
    
    def reset(self):
        self.t = 0
        self.goal_counter = 0
        self.goal_list_xy = []
        angles = []
        angle = np.random.RandomState().uniform(low=0, high=pi/3)
        flip = np.random.RandomState().randint(2)
        angle = -angle if flip == 0 else angle
        rad = np.random.RandomState().uniform(1, 1.5)
        angles.append(angle)
        xy_ = np.array([rad*cos(angle), rad*sin(angle)])
        self.goal_list_xy.append(xy_.copy())
        for _ in range(self.traj_len-1):
            angle = np.random.RandomState().uniform(low=0, high=pi/3)
            flip = np.random.RandomState().randint(2)
            angle = -angle if flip == 0 else angle
            rad = np.random.RandomState().uniform(1, 1.5)
            temp = np.array([rad*cos(angle), rad*sin(angle)])
            xy_ += temp
            self.goal_list_xy.append(xy_.copy())
            angles.append(angle)

        self.goal_list_zeta = angles

        self.goal_list_uv = []
        self.goal_list_r = []
        for i in range(self.traj_len):
            self.goal_list_uv.append([0., 0.])
            self.goal_list_r.append(0.)
        
        xy, zeta, uv, r = self.player.reset()
        #angle = np.random.RandomState().uniform(low=0., high=pi/3)
        #flip = np.random.RandomState().randint(2)
        #zeta = -angle if flip == 0 else angle
        #self.player.angle = degrees(zeta)
        sin_zeta, cos_zeta = sin(zeta), cos(zeta)
        self.set_curr_dists((xy, sin_zeta, cos_zeta, uv, r), np.zeros((2,)))
        self.set_prev_dists((xy, sin_zeta, cos_zeta, uv, r), np.zeros((2,)))
        self.obs = self.get_obs((xy, sin_zeta, cos_zeta, uv, r), np.zeros((2,)))
        return self.obs
    
    def render(self, close=False):
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
            pts = self.player.update_gfx()
            player_pos = [[self.scaling*q+self.WINDOW_SIZE/2. for q in p] for p in pts]
            pygame.gfxdraw.filled_trigon(self.screen, 
                                        int(player_pos[0][0]), int(player_pos[0][1]), 
                                        int(player_pos[1][0]), int(player_pos[1][1]),
                                        int(player_pos[2][0]), int(player_pos[2][1]),
                                        self.player.colour)
            pygame.gfxdraw.filled_circle(self.screen, 
                                        int(0+self.WINDOW_SIZE/2.), 
                                        int(0+self.WINDOW_SIZE/2.), 
                                        int(0.1*self.scaling), 
                                        (0, 255, 0))
            for i, g in enumerate(self.goal_list_xy):
                if i == self.goal_counter: colour = (255,0,0)
                else: colour = (0, 255, 0)
                pygame.gfxdraw.filled_circle(self.screen, 
                                        int(self.scaling*g[0]+self.WINDOW_SIZE/2.), 
                                        int(self.scaling*g[1]+self.WINDOW_SIZE/2.), 
                                        int(0.1*self.scaling), 
                                        colour)
            
            time_text = self.font.render("Timestamp: " + str(self.t)[:8], False, (0,0,0))
            frame_text = self.font.render("Frame: " + str(int(self.t/self.dt)), False, (0,0,0))
            goal_text = self.font.render("Goal index: " + str(self.goal_counter)[:8], False, (0,0,0))
            pos_text = self.font.render("Position: [" + str(self.player.position.x)[:8] + ", " + str(self.player.position.y)[:8] + "]", False, (0,0,0))
            uv_text = self.font.render("Linear velocity: [" + str(self.player.velocity.x)[:8] + ", " + str(self.player.velocity.y)[:8] + "]", False, (0,0,0))
            angle_text = self.font.render("Direction: [" + str(radians(self.player.angle))[:8] + "]", False, (0,0,0))
            r_text = self.font.render("Rotational velocity: [" + str(radians(self.player.angular_velocity))[:8] + "]", False, (0,0,0))
            state_text = self.font.render("Observation vector: " + np.array2string(np.around(np.array(self.obs), 2)), False, (0,0,0))
            action_text = self.font.render("Action: " + np.array2string(self.curr_action), False, (0,0,0))
            

            text_list = [time_text, frame_text, goal_text, pos_text, uv_text, angle_text, r_text, state_text, action_text]
            for i, t in enumerate(text_list):
                self.screen.blit(t, (5, 5 + i* (t.get_height()+5)))

#            goal_pos_text = self.font.render("Goal positions: ", False, (0,0,0))
#            for i, g in enumerate(self.goal_list_xy):
#                for j, h in enumerate(g):
#                    t = self.font.render(h[:3], False, (0,0,0))
#                    self.screen.blit(t, (5+goal_pos_text.get_width()+5+j*(t.get_width()+5), 5+i*(t.get_height()+5)))

            pygame.display.flip()
        #if close==True and self.init:
        #    pygame.display.quit()
        #    pygame.quit()
        #    self.init = False

    def rotate(self, vec, angle):
        cz = cos(angle)
        sz = sin(angle)
        x = vec[0]*cz-vec[1]*sz
        y = vec[0]*sz+vec[1]*cz
        return [x, y]

class Player:
    def __init__(self, x=0., y=0., angle=0., length=0.075, max_steering=45., max_acceleration=7.5, dt=0.05):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0., 0.)
        self.angle = angle
        self.angular_velocity = 0.
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 20
        self.brake_deceleration = 10
        self.free_deceleration = 2
        self.drag_coefficient = 1.
        self.damping_coefficient = 1.

        self.acceleration = 0.
        self.steering = 0.

        self.dt = dt

        self.pts = [[0.3, 0.], [0., 0.1], [0., -0.1]]
        self.colour = (0, 0, 255)

    def step(self, thrust_c, steering_c):
        thrust_c = np.clip(thrust_c, 0., 1.)
        steering_c = np.clip(steering_c, -1., 1.)
        #thrust_c = (thrust_c+0.5)/2
        #print(thrust_c)
        drag = -copysign(self.drag_coefficient*self.velocity.x**2, self.velocity.x)
        #print("drag: ", drag)
        #print("acceleration: ", self.acceleration)
        self.acceleration += (thrust_c*self.max_acceleration+drag)*self.dt
        self.steering += steering_c*self.max_steering*self.dt
        self.velocity += (self.acceleration*self.dt, 0)
        if self.velocity.x < 0: self.velocity.x = 0.
        
        if not self.steering == 0:
            turning_radius = self.length/tan(radians(self.steering))
            self.angular_velocity = self.velocity.x/turning_radius
        else:
            self.angular_velocity = 0.
        
        damping = -copysign(self.drag_coefficient*self.angular_velocity, self.angular_velocity)*self.dt

        self.position += self.velocity.rotate(-self.angle)*self.dt
        self.angle += degrees(self.angular_velocity+damping)*self.dt
        return np.array([self.position.x, self.position.y]), radians(self.angle), np.array([self.velocity.x, self.velocity.y]), self.angular_velocity
    
    def update_gfx(self):
        rad_angle = -radians(self.angle)
        pts = []
        for p in self.pts:
            x_ = self.position.x+(p[0]*cos(rad_angle)-p[1]*sin(rad_angle))
            y_ = self.position.y+(p[0]*sin(rad_angle)+p[1]*cos(rad_angle))
            pts.append([x_, y_])
        return pts

    def reset(self):
        self.position = Vector2(0., 0.)
        self.velocity = Vector2(0., 0.)
        self.angle = 0.
        self.angular_velocity = 0.
        self.acceleration = 0.
        self.steering = 0.
        return np.array([self.position.x, self.position.y]), radians(self.angle), np.array([self.velocity.x, self.velocity.y]), self.angular_velocity