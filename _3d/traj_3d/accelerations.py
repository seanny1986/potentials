import torch
import gym
import gym_aero
import agents as ag
import training_loops as tl
import matplotlib.pyplot as plt

import traj_3d
import soft_3d
import term_3d

"""
Produces time plot of acceleration to show that the proposed method never exceeds a certain level
acceleration. This is our measure of C2 continuity. This is only done for the 3D agents, as the
2D environment is not unstable, and doesn't induce jumping.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_accelerations(env, agent, render=False):
    state = torch.Tensor(env.reset()).to(device)
    reward_sum = 0
    done = False
    t = []
    duvw = []
    dpqr = []
    uvw_prev = state[9:12]
    pqr_prev = state[12:15]
    i = 0
    while not done:
        if render:
            env.render()
        action, _, _, _ = agent.select_action(state.unsqueeze(0))
        action = action.squeeze(0)
        next_state, reward, done, info = env.step(action.cpu().data.numpy())
        reward_sum += reward
        next_state = torch.Tensor(next_state).to(device)
        uvw, pqr = next_state[9:12], next_state[12:15]
        duvw.append(uvw-uvw_prev)
        dpqr.append(pqr-pqr_prev)
        t.append(i)
        state = next_state
        uvw_prev = uvw
        pqr_prev = pqr
        i += 1
    #if render: env.close()
    return reward_sum, duvw, dpqr, t

fname = "./traj_3d/gaussian_2.pth.tar"

t_env = soft_3d.trajectory_env_3d.TrajectoryEnv()

state_dim = t_env.observation_space.shape[0]
action_dim = t_env.action_space.shape[0]
hidden_dim = 256
agent = ag.Agent(state_dim, hidden_dim, action_dim)
print("Agent initialized, loading state dictionary.")
agent.load_state_dict(torch.load("./"+fname, map_location=lambda storage, loc: storage))
print()
print("State dictionary loaded")
reward_sum, duvw, dpqr, t = get_accelerations(t_env, agent)

fig = plt.figure(figsize=(5, 5))

ax = fig.add_subplot(111)
ax.plot(t, duvw)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Linear Acceleration (m/s^2)')

ax2 = fig.add_subplot(111)
ax2.plot(t, dpqr)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Linear Acceleration (m/s^2)')

plt.savefig('./figures/accelerations.png')