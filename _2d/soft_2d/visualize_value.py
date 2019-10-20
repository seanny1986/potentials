import torch
import envs.soft_2d as tenv
import agents.agents as ag
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

#############################################################################################################################################
############################################    VISUALIZATION -- LEARNED GOAL THRESHOLD   ###################################################
#############################################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(hidden_dim=256):
    path = os.getcwd()+"/soft_2d/"
    fname = path+"gaussian_2_term.pth.tar"
    t_env = tenv.TrajectoryEnv2D()

    state_dim = t_env.observation_space.shape[0]
    action_dim = t_env.action_space.shape[0]

    agent = ag.Agent(state_dim, hidden_dim, action_dim)
    print("Agent initialized, loading state dictionary.")
    agent.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    print()
    print("State dictionary loaded")

    xs = np.arange(0, 3, 0.05)
    ys = np.arange(-1.5, 1.5, 0.05)

    X, Y = np.meshgrid(xs, ys)
    grid = np.zeros(X.shape)
    
    theta = np.random.uniform(-pi/2, pi/2)
    x_ng = 1.5*cos(theta)+1.5
    y_ng = 1.5*sin(theta)
    
    T = np.arange(0, 3, 0.1)

    k = 1
    for t in T:
        for j, x in enumerate(xs):
            for k, y in enumerate(ys):
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                arr = np.array([x, y, x_ng, y_ng, 0., 0., t])
                state = torch.Tensor(arr).to(device)
                value = agent.get_integrated_value(state)
                grid[j,k] = value.item()
        ax.pcolormesh(X, Y, grid, cmap="plasma")
        ax.scatter([0, 1.5, x_ng], [0, 0, y_ng], color="k")
        ax.set_xlabel('x position (m)')
        ax.set_ylabel('y position_m')
        plt.savefig('./figures/value_density'+str(k)+'.png')
        print("figure saved")
        k += 1
    