import gym
import gym_aero
import agents.agents as ag
import training_loops.training_loops as tl
import envs.fan_2d as tenv
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(hidden_dim=256):
    path = os.getcwd()+"/_2d/fan_2d/"
    fname = path+"2-wps-0.1.pth.tar"
    t_env = tenv.FanTrajectoryEnv2D()
    state_dim = t_env.observation_space.shape[0]
    action_dim = t_env.action_space.shape[0]
    agent = ag.Agent(state_dim, hidden_dim, action_dim, dim=2)
    print("Agent initialized, loading state dictionary.")
    agent.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    print()
    print("State dictionary loaded")
    k = [tl.test(t_env, agent, render=True) for _ in range(100)]
    rewards = sum(k)/len(k)
    print("Mean reward: ", rewards)