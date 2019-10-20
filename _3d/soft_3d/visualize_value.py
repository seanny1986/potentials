import torch
import envs.waypoint_3d as tenv
import agents.agents as ag
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, exp

#############################################################################################################################################
############################################    VISUALIZATION -- LEARNED GOAL THRESHOLD   ###################################################
#############################################################################################################################################

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
    
    xs = np.linspace(-5, 5, 100)
    ys = np.linspace(-5, 5, 100)
    XS, YS = np.meshgrid(xs, ys)
    VALUE = np.zeros(XS.shape)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            arr = np.array([x, y, 0])
            state = torch.Tensor(arr).to(device)
            value = agent.get_integrated_value(state).item()
            VALUE[i, j] = exp(value)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.pcolormesh(XS, YS, VALUE, cmap="plasma")
    ax.set_xlabel('body x distance (m)')
    ax.set_ylabel('body y distance (m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    plt.savefig('./figures/value_density_wp_3d.png')
    print("figure saved")
    