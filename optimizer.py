import numpy as np
import matplotlib.pyplot as plt

from physics.orbits import pos_earth_moon, orbital_period, compute_L2, optimal_L2_orbit
from physics.dynamics import acceleration
from integrator import evolve

def optimize_delta_r(alpha):
    '''
    Inputs:
        alpha (float) - scaling factor for delta_r calculation. This is what we are optimizing

    Outputs:
        global_error (float) - RMS error between rocket trajectory and optimal L2 orbit
        error (numpy array) - array of position errors at each time step
    '''
    time_step = 10
    T = orbital_period() 
    max_time = 1*T 
    time = np.arange(0, max_time, time_step)

    pos_L2, v_L2 = compute_L2(alpha = alpha)
    r_L2 = optimal_L2_orbit(pos_L2, time)

    r_rocket_barycenter, v_rocket_barycenter = evolve(pos_L2, v_L2, time,
                                                      EoM=acceleration,
                                                      method='RK4')

    errors = np.linalg.norm(r_rocket_barycenter - r_L2, axis=1)

    lengths = np.linalg.norm(r_rocket_barycenter, axis=1)
    rms = np.sqrt(np.mean(errors**2))
    std = np.std(lengths)
    mean = np.mean(lengths)
    return rms

def optimize_step_size(beta):
    '''
    Inputs:
        alpha (float) - scaling factor for delta_r calculation. This is what we are optimizing

    Outputs:
        global_error (float) - RMS error between rocket trajectory and optimal L2 orbit
        error (numpy array) - array of position errors at each time step
    '''
    time_step = beta
    T = orbital_period() 
    max_time = 1*T 
    time = np.arange(0, max_time, time_step)

    pos_L2, v_L2 = compute_L2(alpha = 1.0483849100348734)
    r_L2 = optimal_L2_orbit(pos_L2, time)
    #print("starting evolution!")
    r_rocket_barycenter, v_rocket_barycenter = evolve(pos_L2, v_L2, time, 
                                                      EoM=acceleration,
                                                      method='RK4')
    #print("finished evolution!")
    errors = np.linalg.norm(r_rocket_barycenter - r_L2, axis=1)

    lengths = np.linalg.norm(r_rocket_barycenter, axis=1)
    rms = np.sqrt(np.mean(errors**2))
    std = np.std(lengths)
    mean = np.mean(lengths)
    return rms