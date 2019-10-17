import matplotlib.pyplot as plt
import numpy as np
import agents.agents as ag
import torch
from math import pi
import os

import gym
import gym_aero
import envs.soft_3d as soft_3d
import envs.term_3d as term_3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_test(env, agent, test_name):
    print(test_name)
    print("Init RPM: ", env.get_rpm())
    t_arr, xyz_arr, zeta_arr, uvw_arr, pqr_arr = [], [], [], [], []
    state = torch.Tensor(env.reset()).to(device)
    xyz, zeta, uvw, pqr = env.get_data()
    xyz_arr.append(np.array(xyz).copy())
    zeta_arr.append(np.array(zeta).copy())
    uvw_arr.append(np.array(uvw).copy())
    pqr_arr.append(np.array(pqr).copy())
    
    done = False
    while not done:
        t_arr.append(env.t)
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action.unsqueeze(0))
        xyz, zeta, uvw, pqr = env.get_data()
        xyz_arr.append(np.array(xyz).copy())
        zeta_arr.append(np.array(zeta).copy())
        uvw_arr.append(np.array(uvw).copy())
        pqr_arr.append(np.array(pqr).copy())
        state = torch.Tensor(next_state).to(device)
    t_arr.append(env.t)
    print("Time: ", env.t)
    print("xyz: ", xyz)
    print("zeta: ", zeta)
    print("uvw: ", uvw)
    print("pqr: ", pqr)
    print()
    return t_arr, xyz_arr, zeta_arr, uvw_arr, pqr_arr

def plot_figures(dataset, name):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(48,36))
    fig.suptitle("Gravity Test", fontsize=14)

    ax1.set_title("Inertial X-Position")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")

    ax2.set_title("Inertial Y-Position")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")

    ax3.set_title("Inertial Z-Position")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Position (m)")

    ax4.set_title("Roll Angle")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angle (rad)")

    ax5.set_title("Pitch Angle")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Angle (rad)")

    ax6.set_title("Yaw Angle")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Angle (rad)")

    ax7.set_title("Body U-Velocity")
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Velocity (m/s)")

    ax8.set_title("Body V-Velocity")
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Velocity (m/s)")

    ax9.set_title("Body W-Velocity")
    ax9.set_xlabel("Time (s)")
    ax9.set_ylabel("Velocity (m/s)")

    ax10.set_title("Body P-Velocity")
    ax10.set_xlabel("Time (s)")
    ax10.set_ylabel("Angular Velocity (rad/s)")

    ax11.set_title("Body Q-Velocity")
    ax11.set_xlabel("Time (s)")
    ax11.set_ylabel("Angular Velocity (rad/s)")

    ax12.set_title("Body R-Velocity")
    ax12.set_xlabel("Time (s)")
    ax12.set_ylabel("Angular Velocity (m/s)")
    
    ## RPM response
    
    for d in dataset:
        t, xyz, zeta, uvw, pqr = d
        
        ax1.plot(t, [x[0] for x in xyz])
        ax2.plot(t, [x[1] for x in xyz])
        ax3.plot(t, [x[2] for x in xyz])

        ax4.plot(t, [z[0] for z in zeta])
        ax5.plot(t, [z[1] for z in zeta])
        ax6.plot(t, [z[2] for z in zeta])

        ax7.plot(t, [u[0] for u in uvw])
        ax8.plot(t, [u[1] for u in uvw])
        ax9.plot(t, [u[2] for u in uvw])

        ax10.plot(t, [p[0] for p in pqr])
        ax11.plot(t, [p[1] for p in pqr])
        ax12.plot(t, [p[2] for p in pqr])
    
    ax1.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax2.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax3.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax4.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax5.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax6.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax7.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax8.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax9.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax10.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax11.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    ax12.legend(['Old Simulation, dt=0.01', 'Old Simulation, dt=0.05', 'New Simulation'])
    
    # set axis limits
    ax1.set_ylim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax3.set_ylim(-5, 5)

    ax4.set_ylim(-2*pi, 2*pi)
    ax5.set_ylim(-2*pi, 2*pi)
    ax6.set_ylim(-2*pi, 2*pi)

    ax7.set_ylim(-10, 10)
    ax8.set_ylim(-10, 10)
    ax9.set_ylim(-10, 10)

    plt.show()

    fig.savefig(name+"_state_fig.pdf")

def main():
    traj_arr = []
    path = os.getcwd()+"/traj_3d/"
    fname = path+"gaussian_2_term.pth.tar"
    env = gym.make("Trajectory-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = agent = ag.Agent(state_dim, 512, action_dim, 3)
    print("Agent initialized, loading state dictionary.")
    agent.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    t_arr, xyz_arr, zeta_arr, uvw_arr, pqr_arr = run_test(env, agent, "3D Trajectory Hard Boundary")
    traj_arr.append([t_arr, xyz_arr, zeta_arr, uvw_arr, pqr_arr])

    soft_arr = []
    path = os.getcwd()+"/soft_3d/"
    fname = path+"gaussian_2_term.pth.tar"
    env = soft_3d.TrajectoryEnv3D()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = agent = ag.Agent(state_dim, 512, action_dim, 3)
    print("Agent initialized, loading state dictionary.")
    agent.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    t_arr, xyz_arr, zeta_arr, uvw_arr, pqr_arr = run_test(env, agent, "3D Trajectory Soft Boundary")
    soft_arr.append([t_arr, xyz_arr, zeta_arr, uvw_arr, pqr_arr])

    term_arr = []
    path = os.getcwd()+"/term_3d/"
    fname = path+"gaussian_2_term.pth.tar"
    env = term_3d.TrajectoryEnvTerm()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = agent = ag.Agent(state_dim, 512, action_dim, 3)
    print("Agent initialized, loading state dictionary.")
    t_arr, xyz_arr, zeta_arr, uvw_arr, pqr_arr = run_test(env, agent, "3D Trajectory Learned Termination")
    term_arr.append([t_arr, xyz_arr, zeta_arr, uvw_arr, pqr_arr])

    plot_figures(traj_arr, "traj")
    plot_figures(soft_arr, "soft")
    plot_figures(term_arr, "term")

if __name__ == "__main__":
    main()