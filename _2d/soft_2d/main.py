import torch
import envs.soft_2d as tenv
import agents.agents as ag
import training_loops.training_loops as tl
from common.multiprocessing_env import SubprocVecEnv
import os
import pandas as pd
import config as cfg
import gym

wps = str(cfg.waypoints)
lookahead = cfg.waypoints

def run(num_envs=16, hidden_dim=256, batch_size=1024, iterations=1000, log_interval=10, runs=3):
    envs = [tl.make_soft_2d() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    t_env = tenv.SoftTrajectoryEnv2D()
    #state_size = 10+7*(t_env.num_fut_wp+1)
    #t_env.observation_space = gym.spaces.Box(-1, 1, shape=(state_size,))

    state_dim = t_env.observation_space.shape[0]
    action_dim = t_env.action_space.shape[0]
    path = os.getcwd()+"/_2d/soft_2d/"
    for i in range(runs):
        agent = ag.Agent(state_dim, hidden_dim, action_dim, dim=2, lookahead=lookahead)
        opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
        ep, rew, agent = tl.train_mp(envs, t_env, agent, opt, batch_size, iterations, log_interval, render=True, fname=path+wps+"-wps")
        if i == 0:
            csv_input = pd.DataFrame()
            csv_input["timesteps"] = ep
        csv_input["run"+str(i)] = rew
        csv_input.to_csv(path+"data_wp-"+wps+".csv", index=False)