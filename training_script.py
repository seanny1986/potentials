import waypoint_2d.main as wp_2d_main
import waypoint_3d.main as wp_3d_main
import nh_waypoint_3d.main as nh_wp_3d_main

import traj_2d.main as traj_2d_main
import soft_2d.main as soft_2d_main
import term_2d.main as term_2d_main

import traj_3d.main as traj_3d_main
import soft_3d.main as soft_3d_main
import term_3d.main as term_3d_main

"""
Training script for the agents.
"""

#wp_2d_main.run(num_envs=32)
#wp_3d_main.run(num_envs=32)
#nh_wp_3d_main.run(num_envs=32, hidden_dim=512, iterations=2500)

#traj_2d_main.run()
#soft_2d_main.run()
#term_2d_main.run(runs=1)

#traj_3d_main.run(num_envs=32, hidden_dim=512, iterations=2500)
#soft_3d_main.run(num_envs=32, hidden_dim=512, iterations=2500)
term_3d_main.run(num_envs=32, hidden_dim=512, iterations=2500)