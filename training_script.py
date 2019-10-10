import traj_2d.main as traj_2d_main
import soft_2d.main as soft_2d_main
import term_2d.main as term_2d_main

import traj_3d.main as traj_3d_main
import soft_3d.main as soft_3d_main
import term_3d.main as term_3d_main

"""
Training script for the agents.
"""

#traj_2d_main.run(num_envs=2, batch_size=32, iterations=5, log_interval=1, runs=2)
#soft_2d_main.run(num_envs=2, batch_size=32, iterations=5, log_interval=1, runs=2)
term_2d_main.run(num_envs=2, batch_size=32, iterations=5, log_interval=1, runs=2)

#traj_3d_main.run(num_envs=2, batch_size=32, iterations=5, log_interval=1, runs=2)
#soft_3d_main.run(num_envs=2, batch_size=32, iterations=5, log_interval=1, runs=2)
#term_3d_main.run(num_envs=2, batch_size=32, iterations=5, log_interval=1, runs=2)