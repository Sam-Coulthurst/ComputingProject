import numpy as np
import matplotlib.pyplot as plt

from physics.orbits import pos_earth_moon, orbital_period, compute_L2, optimal_L2_orbit
from physics.dynamics import energy, jacobi_constant, rotate_to_inertial
from integrator import evolve
from plotting import plot_trajectories, animate_trajectories

def optimize_delta_r(alpha):
    '''
    Inputs:
        alpha (float) - scaling factor for delta_r calculation. This is what we are optimizing

    Outputs:
        global_error (float) - RMS error between rocket trajectory and optimal L2 orbit
        error (numpy array) - array of position errors at each time step
    '''
    time_step = 9.442774828081589e-05
    T = orbital_period() 
    max_time = T 
    time = np.arange(0, max_time, time_step)

    pos_L2, v_L2 = compute_L2(alpha = alpha)
    r_L2 = optimal_L2_orbit(pos_L2, v_L2, time)

    r_rocket_barycenter, v_rocket_barycenter = evolve(pos_L2, v_L2, time, time_step, method='RK8')

    error = np.linalg.norm(r_rocket_barycenter - r_L2, axis=0)
    global_error = np.sqrt(np.mean(error**2))
    return error[-1]


def optimize_step_size(alpha):

    time_step = alpha
    T = orbital_period() 
    max_time = T 
    time = np.arange(0, max_time, time_step)

    pos_L2, v_L2 = compute_L2(alpha = 1.0486180339887499)
    r_L2 = optimal_L2_orbit(pos_L2, v_L2, time)

    r_rocket_barycenter, v_rocket_barycenter = evolve(pos_L2, v_L2, time, time_step, method='RK8', frame = 'rotating')
    Jacobi = jacobi_constant(r_rocket_barycenter, v_rocket_barycenter)
    error_Jacobi = (Jacobi - Jacobi[0]) / Jacobi[0]
    global_error = np.sqrt(np.mean(error_Jacobi**2))

    return global_error

