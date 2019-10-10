import gym
import gym_aero
import agents as ag
import training_loops as tl
import gym
import gym_aero
import torch
import numpy as np

#############################################################################################################################################
############################################    VISUALIZATION -- LEARNED GOAL THRESHOLD   ###################################################
#############################################################################################################################################

fname = "gaussian_2_term.pth.tar"

t_env = gym.make("Trajectory-v0")

state_dim = t_env.observation_space.shape[0]
action_dim = t_env.action_space.shape[0]
hidden_dim = 256
agent = ag.Agent(state_dim, hidden_dim, action_dim)
print("Agent initialized, loading state dictionary.")
agent.load_state_dict(torch.load("./"+fname, map_location=lambda storage, loc: storage))
print()
print("State dictionary loaded")
rewards = [tl.test_term(t_env, agent, render=True) for _ in range(100)]
print(np.mean([r[0] for r in rewards]))
print(np.mean([r[1] for r in rewards]))