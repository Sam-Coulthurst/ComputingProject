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
    time_step = 100
    T = orbital_period() 
    max_time = 1*T 
    time = np.arange(0, max_time, time_step)

    pos_L2, v_L2 = compute_L2(alpha = alpha)
    r_L2 = optimal_L2_orbit(pos_L2, time)

    pos_L2 = np.array(pos_L2, dtype=np.float64)
    v_L2 = np.array(v_L2, dtype=np.float64)
    time = np.array(time, dtype=np.float64)
    r_rocket_barycenter, v_rocket_barycenter = evolve(pos_L2, v_L2, time,
                                                      EoM=acceleration,
                                                      method='RK8')

    errors = np.linalg.norm(r_rocket_barycenter - r_L2, axis=1)
    #print(np.shape(errors))
    lengths = np.linalg.norm(r_rocket_barycenter, axis=1)
    rms = np.sqrt(np.mean(errors**2))
    std = np.std(lengths)
    mean = np.mean(lengths)
    plt.plot(r_L2[:,0],r_L2[:,1],linestyle='--', color='black', label='Optimal L2 Orbit')
    plt.plot(r_rocket_barycenter[:,0],r_rocket_barycenter[:,1], label='Rocket Trajectory')
    plt.xlim(-4e8, 4e8)
    plt.ylim(-4e8, 4e8)
    plt.axis('equal')
    return std/mean

def optimize_step_size(beta,meth='RK4'):
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

    pos_L2, v_L2 = compute_L2(alpha = 1.0486155527864045)
    r_L2 = optimal_L2_orbit(pos_L2, time)
    #print("starting evolution!")
    r_rocket_barycenter, v_rocket_barycenter = evolve(pos_L2, v_L2, time, 
                                                      EoM=acceleration,
                                                      method=meth)
    #print("finished evolution!")
    errors = np.linalg.norm(r_rocket_barycenter - r_L2, axis=1)

    lengths = np.linalg.norm(r_rocket_barycenter, axis=1)
    rms = np.sqrt(np.mean(errors**2))
    std = np.std(lengths)
    mean = np.mean(lengths)
    return std/mean