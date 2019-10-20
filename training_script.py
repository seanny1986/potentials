import _2d.waypoint_2d.main as wp_2d_main
import _2d.traj_2d.main as traj_2d_main
import _2d.soft_2d.main as soft_2d_main
import _2d.term_2d.main as term_2d_main

import _3d.waypoint_3d.main as wp_3d_main
import _3d.nh_waypoint_3d.main as nh_wp_3d_main
import _3d.traj_3d.main as traj_3d_main
import _3d.soft_3d.main as soft_3d_main
import _3d.term_3d.main as term_3d_main

import config as cfg

"""
Training script for the agents.
"""

envs = cfg.num_envs
hd = cfg.hidden_dim
bs = cfg.batch_size
iters = cfg.iterations
li = cfg.log_interval
r = cfg.runs

#wp_2d_main.run(num_envs=envs, hidden_dim=hd, batch_size=bs, iterations=iters, log_interval=li, runs=r)
#traj_2d_main.run(num_envs=envs, hidden_dim=hd, batch_size=bs, iterations=iters, log_interval=li, runs=r)
#soft_2d_main.run(num_envs=envs, hidden_dim=hd, batch_size=bs, iterations=iters, log_interval=li, runs=r)
#term_2d_main.run(num_envs=envs, hidden_dim=hd, batch_size=bs, iterations=iters, log_interval=li, runs=r)

#wp_3d_main.run(num_envs=envs, hidden_dim=hd, batch_size=bs, iterations=iters, log_interval=li, runs=r)
#nh_wp_3d_main.run(num_envs=envs, hidden_dim=hd, batch_size=bs, iterations=iters, log_interval=li, runs=r)
traj_3d_main.run(num_envs=envs, hidden_dim=hd, batch_size=bs, iterations=iters, log_interval=li, runs=r)
soft_3d_main.run(num_envs=envs, hidden_dim=hd, batch_size=bs, iterations=iters, log_interval=li, runs=r)
term_3d_main.run(num_envs=envs, hidden_dim=hd, batch_size=bs, iterations=iters, log_interval=li, runs=r)