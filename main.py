import numpy as np
from physics.orbits import pos_earth_moon, orbital_period, compute_L2
from integrator import evolve
from plotting import plot_trajectories

# Time Parameters
T = orbital_period() #s
time_step = 10 #s
max_time = 2 * T #s
time = np.arange(0, max_time, time_step) #s

# Initial conditions - L2 point
pos_L2, v_L2 = compute_L2()

# Calculate trajectory
r_rocket_barycenter, v_rocket_barycenter = evolve(pos_L2, v_L2, time, time_step, method='RK4')
print(r_rocket_barycenter)
pos_earth, pos_moon = pos_earth_moon(time)

#Plot the graph
plot_trajectories(pos_earth, pos_moon, r_rocket_barycenter)
