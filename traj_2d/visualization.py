import agents.agents as ag
import training_loops.training_loops as tl
import envs.traj_2d as tenv
import torch

#############################################################################################################################################
############################################    VISUALIZATION -- LEARNED GOAL THRESHOLD   ###################################################
#############################################################################################################################################

fname = "gaussian_2_term.pth.tar"
t_env = tenv.TrajectoryEnv2D()
state_dim = t_env.observation_space.shape[0]
action_dim = t_env.action_space.shape[0]
hidden_dim = 256
agent = ag.Agent(state_dim, hidden_dim, action_dim)
print("Agent initialized, loading state dictionary.")
agent.load_state_dict(torch.load("./"+fname, map_location=lambda storage, loc: storage))
print()
print("State dictionary loaded")
rewards = [tl.test(t_env, agent, render=True) for _ in range(100)]
print(sum(rewards)/len(rewards))