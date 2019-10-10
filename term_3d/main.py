import torch
import envs.term_3d as tenv
import agents.term_agents as ag
import training_loops.training_loops as tl
from common.multiprocessing_env import SubprocVecEnv
import os
import pandas as pd

def run(num_envs=16, hidden_dim=256, batch_size=1024, iterations=1000, log_interval=10, runs=5):
    envs = [tl.make_term_3d() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    t_env = tenv.TrajectoryEnvTerm()
    state_dim = t_env.observation_space.shape[0]
    action_dim = t_env.action_space.shape[0]
    path = os.getcwd()+"/term_3d/"
    for i in range(runs):
        agent = ag.Agent(state_dim, hidden_dim, action_dim)
        opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
        ep, rew, agent = tl.train_mp(envs, t_env, agent, opt, batch_size, iterations, log_interval, render=False, fname="gaussian_"+str(2))
        if i == 0:
            csv_input = pd.DataFrame()
            csv_input["timesteps"] = ep
        csv_input["run"+str(i)] = rew
        csv_input.to_csv(path+"data.csv", index=False)