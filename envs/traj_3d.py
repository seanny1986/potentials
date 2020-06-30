import gym
import numpy as np
from math import pi, sin, cos, acos, tanh, exp, sqrt
from scipy import interpolate
from gym_aero.envs import env_base
import envs.env_config as cfg

class TrajectoryEnv3D(env_base.AeroEnv):
    def __init__(self):
        super(TrajectoryEnv3D, self).__init__()
        self.name = "Trajectory-v0"
        
        self.goal_rad = 1.5
        self.traj_len = 7
        self.goal_thresh = 0.1
        self.max_dist = 5
        self.T = 3.5
        self._max_episode_steps = int(self.T/self.ctrl_dt)

        self.epsilon_time = 1.5
        self.num_fut_wp = 2
        num_states = 22 + 15 * (self.num_fut_wp + 1)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(num_states,))

        self.temperature = 150
        self.pdf_norm = 1 / sqrt(pi / self.temperature)
        self.waypoint_dist_upper_bound = cfg.waypoint_dist_upper_bound
        self.waypoint_dist_lower_bound = cfg.waypoint_dist_lower_bound
        self.max_spread = cfg.max_spread
        print("Trajectory Env 3D initialized.")
    
    def switch_goal(self, state):
        if self.curr_dist < self.goal_thresh: return True
        else: return False
    
    def get_goal_positions(self):
        n = int(3*(self.num_fut_wp+1))
        return self.obs[:n]

    def term_reward(self, term):
        if term: return 100
        else: return 0.

    def reward(self, state, action, normalized_rpm):
        xyz, sin_zeta, cos_zeta, xyz_dot, pqr = state

        # agent gets a negative reward based on how far away it is from the desired goal state
        dist_rew = 100*(self.prev_dist-self.curr_dist)
        sin_att_rew = 0*(self.prev_att_sin-self.curr_att_sin)
        cos_att_rew = 0*(self.prev_att_cos-self.curr_att_cos)
        vel_rew = 1*(self.prev_vel-self.curr_vel)
        ang_rew = 0.1*(self.prev_ang-self.curr_ang)
        uvw = self.inertial_to_body(xyz_dot)
        heading_rew = -10*uvw[1]**2  # negative reward component for movement in y-direction
        forward_rew = 10*uvw[0]/abs(uvw[0])

        att_rew = sin_att_rew+cos_att_rew

        # agent gets a negative reward for excessive action inputs
        ctrl_rew = -0*sum([((a-self.hov_rpm)/self.max_rpm)**2 for a in action])

        # derivative rewards
        ctrl_dev_rew = -0*sum([(a-b)**2 for a,b in zip(action, self.prev_action)])
        dist_dev_rew = -0*sum([(x-y)**2 for x, y in zip(xyz, self.prev_xyz)])
        sin_att_dev_rew = -0*sum([(sz-sin(z))**2 for sz, z in zip(sin_zeta, self.prev_zeta)])
        cos_att_dev_rew = -0*sum([(cz-cos(z))**2 for cz, z in zip(cos_zeta, self.prev_zeta)])
        vel_dev_rew = -0*sum([(u-v)**2 for u,v in zip(xyz_dot, self.prev_xyz_dot)])
        ang_dev_rew = -0*sum([(p-q)**2 for p, q in zip(pqr, self.prev_pqr)])
        
        ctrl_rew += ctrl_dev_rew+vel_dev_rew+ang_dev_rew+dist_dev_rew+sin_att_dev_rew+cos_att_dev_rew

        # time reward to incentivize using the full time period
        time_rew = 0

        # calculate total reward
        total_reward = dist_rew + att_rew + vel_rew + ang_rew + ctrl_rew + time_rew + heading_rew + forward_rew

        return total_reward, {"dist_rew": dist_rew,
                                "att_rew": att_rew,
                                "vel_rew": vel_rew,
                                "ang_rew": ang_rew,
                                "ctrl_rew": ctrl_rew,
                                "ctrl_dev_rew": ctrl_dev_rew,
                                "dist_dev_rew": dist_dev_rew,
                                "att_dev_rew": sin_att_dev_rew + cos_att_dev_rew,
                                "uvw_dev_rew": vel_dev_rew,
                                "pqr_dev_rew": ang_dev_rew,
                                "heading_rew": heading_rew,
                                "time_rew": time_rew}
    
    def get_obs(self, state, action, normalized_rpm):
        xyz, sin_zeta, cos_zeta, xyz_dot, pqr = state
        xyz_obs = []
        sin_zeta_obs = []
        cos_zeta_obs = []
        xyz_dot_obs = []
        pqr_obs = []
        for i in range(self.num_fut_wp+1):
            if self.goal_counter + i <= len(self.goal_list_xyz) - 1:
                xyz_obs = xyz_obs + self.inertial_to_body([x - g for x,g in zip(xyz, self.goal_list_xyz[self.goal_counter + i])])
                sin_zeta_obs = sin_zeta_obs + [z - sin(g) for z,g in zip(sin_zeta, self.goal_list_zeta[self.goal_counter + i])]
                cos_zeta_obs = cos_zeta_obs + [z - cos(g) for z,g in zip(cos_zeta, self.goal_list_zeta[self.goal_counter + i])]
                xyz_dot_obs = xyz_dot_obs + self.inertial_to_body([u - g for u,g in zip(xyz_dot, self.goal_list_xyz_dot[self.goal_counter + i])])
                pqr_obs = pqr_obs + [p - g for p,g in zip(pqr, self.goal_list_pqr[self.goal_counter + i])]
            else:
                xyz_obs = xyz_obs + [0., 0., 0.]
                sin_zeta_obs = sin_zeta_obs + [0., 0., 0.]
                cos_zeta_obs = cos_zeta_obs + [0., 0., 0.]
                xyz_dot_obs = xyz_dot_obs + [0., 0., 0.]
                pqr_obs = pqr_obs + [0., 0., 0.]     
        
        # derivatives
        duvw_dt = self.inertial_to_body([(u-v)/self.dt for u, v in zip(xyz_dot, self.prev_xyz_dot)])
        dpqr_dt = [(p-q)/self.dt for p, q in zip(pqr, self.prev_pqr)]
        derivatives = self.inertial_to_body(xyz_dot) + duvw_dt + dpqr_dt

        # actions
        da_dt = [(a-b)/self.dt for a, b in zip(action, self.prev_action)]

        tar_obs = xyz_obs + sin_zeta_obs + cos_zeta_obs + xyz_dot_obs + pqr_obs + derivatives
        next_state = tar_obs + action + da_dt + normalized_rpm + [self.t]
        return next_state
    
    def set_curr_dists(self, state, action, normalized_rpm):
        xyz, sin_zeta, cos_zeta, xyz_dot, pqr = state
        if not self.goal_counter > self.traj_len-1:
            self.curr_dist = np.linalg.norm([x - g for x, g in zip(xyz, self.goal_list_xyz[self.goal_counter])])
            self.curr_att_sin = np.linalg.norm([sz - sin(g) for sz, g in zip(sin_zeta, self.goal_list_zeta[self.goal_counter])])
            self.curr_att_cos = np.linalg.norm([cz - cos(g) for cz, g in zip(cos_zeta, self.goal_list_zeta[self.goal_counter])])
            self.curr_vel = np.linalg.norm([x - g for x, g in zip(xyz_dot, self.goal_list_xyz_dot[self.goal_counter])])
            self.curr_ang = np.linalg.norm([x - g for x, g in zip(pqr, self.goal_list_pqr[self.goal_counter])])
        _, zeta, _, _ = self.get_data()
        self.curr_xyz = xyz
        self.curr_zeta = zeta
        self.curr_xyz_dot = xyz_dot
        self.curr_pqr = pqr
        self.curr_action = action

    def step(self, action):
        commanded_rpm = self.translate_action(action)
        self.iris.sim_step(commanded_rpm[0], commanded_rpm[1], commanded_rpm[2], commanded_rpm[3], self.sim_steps)
        xyz, zeta, uvw, pqr = self.get_data()
        xyz_dot = self.get_xyz_dot()
        sin_zeta = [sin(z) for z in zeta]
        cos_zeta = [cos(z) for z in zeta]
        current_rpm = self.get_rpm()
        normalized_rpm = [rpm/self.max_rpm for rpm in current_rpm]
        self.t += self.dt
        self.set_curr_dists((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), commanded_rpm, normalized_rpm)
        reward, info = self.reward((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), commanded_rpm, normalized_rpm)
        term = self.switch_goal((xyz, sin_zeta, cos_zeta, xyz_dot, pqr))
        if term:
            if self.goal_counter < self.traj_len-1:
                term_rew = self.term_reward(True)
                self.goal_counter += 1
                self.set_curr_dists((xyz, sin_zeta, cos_zeta, uvw, pqr), commanded_rpm, normalized_rpm)
                self.t = 0
            else: term_rew = self.term_reward(False)
        else:
            term_rew = self.term_reward(False)
        reward += term_rew
        done = self.terminal((xyz, sin_zeta, cos_zeta, xyz_dot, pqr))
        obs = self.get_obs((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), commanded_rpm, normalized_rpm)
        self.set_prev_dists((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), commanded_rpm)
        info.update({"term_rew" : term_rew})
        return obs, reward, done, info

    def reset(self):
        self.t = 0
        self.goal_counter = 0
        self.goal_list_xyz = []
        xyz_ = [0., 0., 0.]
        for _ in range(self.traj_len):
            temp = self.generate_waypoint()
            xyz_ = [x+g for x, g in zip(xyz_, temp)]
            self.goal_list_xyz.append(xyz_)
        self.goal_list_zeta = []
        self.goal_list_xyz_dot = []
        self.goal_list_pqr = []
        for _ in range(self.traj_len):
            self.goal_list_zeta.append([0., 0., 0.])
            self.goal_list_xyz_dot.append([0., 0., 0.])
            self.goal_list_pqr.append([0., 0., 0.])
        state = super(TrajectoryEnv3D, self).reset()
        xyz, sin_zeta, cos_zeta, uvw, pqr, normalized_rpm = state
        xyz_dot = self.get_xyz_dot()
        self.set_curr_dists((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), self.hov_rpm_, normalized_rpm)
        self.set_prev_dists((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), self.hov_rpm_)
        self.obs = self.get_obs((xyz, sin_zeta, cos_zeta, xyz_dot, pqr), self.hov_rpm_, normalized_rpm)
        return self.obs
    
    def generate_waypoint(self):
        phi = np.random.RandomState().uniform(low=-pi/3, high=pi/3)
        theta = np.random.RandomState().uniform(low=-pi/3, high=pi/3)
        rad = np.random.RandomState().uniform(low=1, high=2.5)
        y = rad*sin(theta)*cos(phi)
        x = rad*cos(theta)*cos(phi)
        z = -rad*sin(theta)
        return [x, y, z]
    
    def render(self, mode='human', video=False, close=False):
        super(TrajectoryEnv3D, self).render(mode=mode, close=close)
        self.ani.draw_goal(np.zeros((3,))) 
        for g in self.goal_list_xyz:
            if np.array_equal(g, self.goal_list_xyz[self.goal_counter]):
                self.ani.draw_goal(g, color=(1., 0., 0.))
            else: 
                self.ani.draw_goal(g) 
        for i, g in enumerate([np.zeros((3,))]+self.goal_list_xyz): 
            if i <= len(self.goal_list_xyz)-1:
                self.ani.draw_line(g, self.goal_list_xyz[i])
        
        # draw goal and thrust vectors
        #self.ani.draw_vector(self.body_to_inertial([self.obs[0:3]]), [0, 0, 0])
        #self.ani.draw_vector(self.body_to_inertial([self.obs[3:6]]), [0, 0, 0])
        #self.ani.draw_vector(self.body_to_inertial([self.obs[6:9]]), [0, 0, 0])
        
        # render state text
        """
        self.ani.draw_label("Frame: {0:.2f}".format(self.t/self.ctrl_dt), (self.ani.window.width // 2, 30.0))
        self.ani.draw_label("Goal index: {}".format(self.goal_counter), (self.ani.window.width // 2, 40.0))
        self.ani.draw_label("Current Goal Position: [{:.2f}, {:.2f}, {:.2f}]".format(self.goal_list_xyz[self.goal_counter][0], 
                                                                                     self.goal_list_xyz[self.goal_counter][1], 
                                                                                     self.goal_list_xyz[self.goal_counter][2]),
                                                                                     (self.ani.window.width // 2, 20.0))

        self.ani.draw_label("-------- State --------", (self.ani.window.width // 2, 50.0))
        self.ani.draw_label("Thrust: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.curr_action[0],
                                                                              self.curr_action[1],
                                                                              self.curr_action[2],
                                                                              self.curr_action[3]), 
                                                                              (self.ani.window.width // 2, 60.0))
        self.ani.draw_label("Position: [{:.2f}, {:.2f}, {:.2f}]".format(self.curr_xyz[0], 
                                                                        self.curr_xyz[1], 
                                                                        self.curr_xyz[2]), 
                                                                        (self.ani.window.width // 2, 70.0))
        self.ani.draw_label("Linear Velocity: [{:.2f}, {:.2f}, {:.2f}]".format(self.curr_xyz_dot[0], 
                                                                               self.curr_xyz_dot[1], 
                                                                               self.curr_xyz_dot[2]), 
                                                                               (self.ani.window.width // 2, 80.0))
        self.ani.draw_label("Linear Acceleration: [{:.2f}, {:.2f}, {:.2f}]".format(self.curr_xyz_dot[0], 
                                                                               self.curr_xyz_dot[1], 
                                                                               self.curr_xyz_dot[2]), 
                                                                               (self.ani.window.width // 2, 90.0))
        self.ani.draw_label("Zeta: [{:.2f}, {:.2f}, {:.2f}]".format(self.curr_xyz_dot[0], 
                                                                    self.curr_xyz_dot[1], 
                                                                    self.curr_xyz_dot[2]), 
                                                                    (self.ani.window.width // 2, 100.0))
        self.ani.draw_label("Angular Velocity: [{:.2f}, {:.2f}, {:.2f}]".format(self.curr_xyz_dot[0], 
                                                                               self.curr_xyz_dot[1], 
                                                                               self.curr_xyz_dot[2]), 
                                                                               (self.ani.window.width // 2, 110.0))
        self.ani.draw_label("Tranisition PDF: [{:.2f}, {:.2f}, {:.2f}]".format(self.curr_xyz_dot[0], 
                                                                               self.curr_xyz_dot[1], 
                                                                               self.curr_xyz_dot[2]), 
                                                                               (self.ani.window.width // 2, 120.0))
        
        self.ani.draw_label("-------- Observation --------", (self.ani.window.width // 2, 20.0))
        self.ani.draw_label("Position Vector: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[0],
                                                                                                                                self.obs[1],
                                                                                                                                self.obs[2],
                                                                                                                                self.obs[3],
                                                                                                                                self.obs[4],
                                                                                                                                self.obs[5],
                                                                                                                                self.obs[6],
                                                                                                                                self.obs[7],
                                                                                                                                self.obs[8]), 
                                                                                                                                (self.ani.window.width // 2, 130.0))
        self.ani.draw_label("Zeta Vector (sin): [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[9],
                                                                                                                                self.obs[10],
                                                                                                                                self.obs[11],
                                                                                                                                self.obs[12],
                                                                                                                                self.obs[13],
                                                                                                                                self.obs[14],
                                                                                                                                self.obs[15],
                                                                                                                                self.obs[16],
                                                                                                                                self.obs[17]), 
                                                                                                                                (self.ani.window.width // 2, 140.0))
        self.ani.draw_label("Zeta Vector (cos): [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[18],
                                                                                                                                self.obs[19],
                                                                                                                                self.obs[20],
                                                                                                                                self.obs[21],
                                                                                                                                self.obs[22],
                                                                                                                                self.obs[23],
                                                                                                                                self.obs[24],
                                                                                                                                self.obs[25],
                                                                                                                                self.obs[26]), 
                                                                                                                                (self.ani.window.width // 2, 150.0))
        self.ani.draw_label("Velocity Vector: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[27],
                                                                                                                                self.obs[28],
                                                                                                                                self.obs[29],
                                                                                                                                self.obs[30],
                                                                                                                                self.obs[31],
                                                                                                                                self.obs[32],
                                                                                                                                self.obs[33],
                                                                                                                                self.obs[34],
                                                                                                                                self.obs[35]), 
                                                                                                                                (self.ani.window.width // 2, 160.0))
        self.ani.draw_label("Omega Vector: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[36],
                                                                                                                            self.obs[37],
                                                                                                                            self.obs[38],
                                                                                                                            self.obs[39],
                                                                                                                            self.obs[40],
                                                                                                                            self.obs[41],
                                                                                                                            self.obs[42],
                                                                                                                            self.obs[43],
                                                                                                                            self.obs[44]), 
                                                                                                                            (self.ani.window.width // 2, 170.0))
        self.ani.draw_label("Derivatives -- UVW: [{:.2f}, {:.2f}, {:.2f}]".format(self.obs[45],
                                                                                    self.obs[46],
                                                                                    self.obs[47]), 
                                                                                    (self.ani.window.width // 2, 180.0))
        self.ani.draw_label("Derivatives -- dUVW_dt: [{:.2f}, {:.2f}, {:.2f}]".format(self.obs[48],
                                                                                    self.obs[49],
                                                                                    self.obs[50]), 
                                                                                    (self.ani.window.width // 2, 190.0))
        self.ani.draw_label("Derivatives -- dPQR_dt: [{:.2f}, {:.2f}, {:.2f}]".format(self.obs[51],
                                                                                    self.obs[52],
                                                                                    self.obs[53]), 
                                                                                    (self.ani.window.width // 2, 200.0))
        self.ani.draw_label("Action: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[54],
                                                                            self.obs[55],
                                                                            self.obs[56],
                                                                            self.obs[57]), 
                                                                            (self.ani.window.width // 2, 210.0))
        self.ani.draw_label("Action Derivative: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[58],
                                                                                        self.obs[59],
                                                                                        self.obs[60],
                                                                                        self.obs[61]), 
                                                                                        (self.ani.window.width // 2, 220.0))
        self.ani.draw_label("Current RPM: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.obs[62],
                                                                                    self.obs[63],
                                                                                    self.obs[64],
                                                                                    self.obs[65]), 
                                                                                    (self.ani.window.width // 2, 230.0))
        self.ani.draw_label("Time: {0:.2f}".format(self.obs[66]), (self.ani.window.width // 2, 240.0))
        """
        self.ani.draw()
        if video: self.ani.save_frame("Trajectory")
        if close:
            self.ani.close_window()
            self.init_rendering = False