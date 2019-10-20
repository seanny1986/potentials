import gym
import gym_aero
import agents.agents as ag
import training_loops.training_loops as tl
import envs.waypoint_3d as tenv
import torch
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(env, agent):
    state = torch.Tensor(env.reset()).to(device)
    uvws, pqrs, ts = [], [], []
    t = 0
    done = False
    while not done:
        action, _, _, _ = agent.select_action(state.unsqueeze(0))
        action = action.squeeze(0)

        uvw = env.curr_uvw
        pqr = env.curr_pqr
        uvws.append(uvw)
        pqrs.append(pqr)
        ts.append(t)
    
        next_state, _, done, _ = env.step(action.cpu().data.numpy())
        next_state = torch.Tensor(next_state).to(device)
        state = next_state
        t += 1
    return uvws, pqrs, ts

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
    uvws, pqrs, ts = test(t_env, agent)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(ts, uvws)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('u (m/s)')
    ax.set_xlim([0, 3])
    ax.set_ylim([-1.5, 1.5])
    plt.savefig('./figures/value_density.png')
    print("figure saved")