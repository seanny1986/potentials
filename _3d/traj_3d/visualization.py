import gym
import gym_aero
import agents.agents as ag
import training_loops.training_loops as tl
import torch
import numpy as np
import os
import envs.traj_3d as tenv

#############################################################################################################################################
############################################    VISUALIZATION -- LEARNED GOAL THRESHOLD   ###################################################
#############################################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(hidden_dim=512):
    path = os.getcwd()+"/_3d/traj_3d/"
    fname = path+"2-wps.pth.tar"
    t_env = tenv.TrajectoryEnv3D()
    state_dim = t_env.observation_space.shape[0]
    action_dim = t_env.action_space.shape[0]
    agent = ag.Agent(state_dim, hidden_dim, action_dim, 3)
    print("Agent initialized, loading state dictionary.")
    agent.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    print()
    print("State dictionary loaded")
    tup = [tl.test(t_env, agent, render=True) for _ in range(100)]
    k = [t[0] for t in tup]
    rewards = sum(k)/len(k)
    print("Mean reward: ", rewards)