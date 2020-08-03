import _2d.waypoint_2d.visualization as wp_2d_main
import _2d.fan_2d.visualization as fan_2d_main
import _2d.traj_2d.visualization as traj_2d_main
import _2d.soft_2d.visualization as soft_2d_main
import _2d.term_2d.visualization as term_2d_main

import _3d.waypoint_3d.visualization as wp_3d_main
import _3d.traj_3d.visualization as traj_3d_main
import _3d.soft_3d.visualization as soft_3d_main
import _3d.term_3d.visualization as term_3d_main


#wp_2d_main.run()
#wp_3d_main.run()
#nh_wp_3d_main.run()

fan_2d_main.run(hidden_dim=512)
#traj_2d_main.run(hidden_dim=512)
#soft_2d_main.run(hidden_dim=256)
#term_2d_main.run(hidden_dim=256)

#traj_3d_main.run()
soft_3d_main.run()
#term_3d_main.run()