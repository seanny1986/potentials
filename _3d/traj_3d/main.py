import torch
import gym
import gym_aero
import envs.traj_3d as tenv
import agents.agents as ag
import training_loops.training_loops as tl
from common.multiprocessing_env import SubprocVecEnv
import os
import pandas as pd
import config as cfg
import envs.env_config as ecfg

wps = str(ecfg.num_fut_wp)

def run(logger, num_envs=16, hidden_dim=256, batch_size=1024, iterations=1000, log_interval=10, runs=3, t_runs=10):
    envs = [tl.make_traj_3d() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    t_env = tenv.TrajectoryEnv3D()
    state_dim = t_env.observation_space.shape[0]
    action_dim = t_env.action_space.shape[0]
    path = os.getcwd()+"/_3d/traj_3d/"
    logger.info(" ")
    logger.info("Initializing Hard Goals 3D Waypoint Navigation Environment")
    logger.info("Working directory is " + path)
    logger.info(" ")
    logger.info("----------------------------------------------")
    logger.info("Experiment Settings")
    logger.info("----------------------------------------------")
    logger.info("Number of workers: {}".format(num_envs))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("Iterations: {}".format(iterations))
    logger.info("Log interval: {}".format(log_interval))
    logger.info("Experiment runs: {}".format(iterations))
    logger.info("Number of test runs: {}".format(t_runs))
    logger.info("----------------------------------------------")
    logger.info("Network Parameters")
    logger.info("----------------------------------------------")
    logger.info("Network input dim: {}".format(state_dim))
    logger.info("Network hidden dim: {}".format(hidden_dim))
    logger.info("Network layers: {}".format(2))
    logger.info("Network output dim: {}".format(action_dim))
    logger.info("----------------------------------------------")
    logger.info("Environment Parameters")
    logger.info("----------------------------------------------")
    logger.info("Trajectory length: {}".format(t_env.traj_len))
    logger.info("Obs future waypoints: {}".format(t_env.num_fut_wp))
    logger.info("Distribution temperature: {}".format(t_env.temperature))
    logger.info("Waypoint distance bounds: {}, {}".format(t_env.waypoint_dist_lower_bound, t_env.waypoint_dist_upper_bound))
    logger.info("Maximum spread: {}".format(t_env.max_spread))
    for i in range(runs):
        agent = ag.Agent(state_dim, hidden_dim, action_dim, dim=3)
        opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
        ep, rew, agent = tl.train_mp(logger, envs, t_env, agent, opt, batch_size, iterations, log_interval, t_runs, render=False, fname=path+wps+"-wps")
        if i == 0:
            csv_input = pd.DataFrame()
            csv_input["timesteps"] = ep
        csv_input["run"+str(i)] = rew
        csv_input.to_csv(path+"data_wp-"+wps+".csv", index=False)