import agents.agents as ag
import training_loops.training_loops as tl
import envs.traj_2d as tenv
import torch

import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt

#############################################################################################################################################
############################################    VISUALIZATION -- LEARNED GOAL THRESHOLD   ###################################################
#############################################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(fname, hidden_dim=256):
    t_env = tenv.TrajectoryEnv2D()

    state_dim = t_env.observation_space.shape[0]
    action_dim = t_env.action_space.shape[0]

    agent = ag.Agent(state_dim, hidden_dim, action_dim)
    print("Agent initialized, loading state dictionary.")
    agent.load_state_dict(torch.load("./"+fname, map_location=lambda storage, loc: storage))
    print()
    print("State dictionary loaded")

    xs = np.arange(-5,5,0.1)
    ys = np.arange(-5,5,0.1)

    X, Y = np.meshgrid(xs, ys)
    grid = torch.zeros(xs.shape[0], ys.shape[0])
    for i in range(50):
        for j, X in enumerate(xs):
            for k, Y in enumerate(ys):
                theta = np.random.uniform(-pi/2, pi/2)
                x = 1.5*sin(theta)+1.5
                y = 1.5*cos(theta)
                arr = np.array([X, Y, 1.5, 0, x, y])
                state = torch.Tensor(arr).to(device)
                value = agent.get_2d_value(state)
                grid[j,k] = value

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(1.5, 0)
    ax.pcolormesh(X, Y, grid, cmap=plt.cm.Greens_r)
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position_m')
    plt.savefig('./figures/value_density.png')