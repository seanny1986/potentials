import torch
import trajectory_env_2d as tenv
import agents as ag
import training_loops as tl

"""
Generates trajectories of arbitrary length, and test the completion rate of the policy. Ideally,
it should reach a very high completion rate.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_term(env, agent, render=False):
    state = torch.Tensor(env.reset()).to(device)
    reward_sum = 0
    done = False
    while not done:
        action, _, _, _ = agent.select_action(state.unsqueeze(0))
        action = action.squeeze(0)
        next_state, reward, done, info = env.step(action.cpu().data.numpy())
        reward_sum += reward
        next_state = torch.Tensor(next_state).to(device)
        state = next_state
    return reward_sum

fname = "gaussian_2.pth.tar"

t_env = tenv.TrajectoryEnv2D()

state_dim = t_env.observation_space.shape[0]
action_dim = t_env.action_space.shape[0]
hidden_dim = 256
agent = ag.Agent(state_dim, hidden_dim, action_dim)
print("Agent initialized, loading state dictionary.")
agent.load_state_dict(torch.load("./"+fname, map_location=lambda storage, loc: storage))
print()
print("State dictionary loaded")
rewards = np.mean([tl.test_term(t_env, agent, render=True) for _ in range(100)])