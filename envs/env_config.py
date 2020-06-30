from math import pi

##################
# 2D ENVIRONMENTS#
##################
DIM = 15
WINDOW_SIZE = 1000
dt = 0.05 
length = 0.3 
max_steering_angle = 30 
max_thrust = 7.5
max_acceleration = 5.
max_velocity = 3
tau_thrust = 0.8
tau_steering = 0.5
mass = 1.
cd = 1.

# init parameters
x = 0
y = 0
u = 0
v = 0
angle = 0
angular_velocity = 0
acceleration = 0
steering_angle = 0
thrust = 0
steering = 0
drag = 0

goal_thresh = 0.1
temperature = 150
max_spread = pi/2
subtraj_time = 3
traj_len = 5
num_fut_wp = 2
waypoint_dist_upper_bound = 2.5
waypoint_dist_lower_bound = 1


##################
# 3D ENVIRONMENTS#
##################
