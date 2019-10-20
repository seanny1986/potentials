import gym
import gym_aero
import agents.agents as ag
import training_loops.training_loops as tl
import envs.waypoint_3d as tenv
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(hidden_dim=256):
    path = os.getcwd()+"/waypoint_3d/"
    fname = path+"gaussian_2_term.pth.tar"
    t_env = tenv.WaypointEnv3D()
    state_dim = t_env.observation_space.shape[0]
    action_dim = t_env.action_space.shape[0]
    agent = ag.Agent(state_dim, hidden_dim, action_dim, dim=3)
    print("Agent initialized, loading state dictionary.")
    agent.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    print()
    print("State dictionary loaded")
    k = [tl.test(t_env, agent, render=True) for _ in range(100)]
    rewards = sum(k)/len(k)
    print("Mean reward: ", rewards)