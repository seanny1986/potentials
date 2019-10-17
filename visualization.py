import waypoint_2d.visualization as wp_2d_main
import waypoint_3d.visualization as wp_3d_main
import nh_waypoint_3d.visualization as nh_wp_3d_main

import traj_2d.visualization as traj_2d_main
import soft_2d.visualization as soft_2d_main
#import term_2d.main as term_2d_main

import traj_3d.visualization as traj_3d_main
#import soft_3d.main as soft_3d_main
#import term_3d.main as term_3d_main


#wp_2d_main.run()
#wp_3d_main.run()
nh_wp_3d_main.run(hidden_dim=512)

#traj_2d_main.run()
#soft_2d_main.run()

#traj_3d_main.run(hidden_dim=512)