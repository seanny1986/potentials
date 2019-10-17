import waypoint_2d.visualize_value as wp_2d_main
import waypoint_3d.visualize_value as wp_3d_main
import nh_waypoint_3d.visualize_value as nh_wp_3d_main

import traj_2d.visualize_value as traj_2d_main
import soft_2d.visualize_value as soft_2d_main
#import term_2d.main as term_2d_main

#import traj_3d.main as traj_3d_main
#import soft_3d.main as soft_3d_main
#import term_3d.main as term_3d_main

#wp_2d_main.run(hidden_dim=256)
#wp_3d_main.run(hidden_dim=256)
nh_wp_3d_main.run(hidden_dim=512)

#traj_2d_main.run()
#soft_2d_main.run()