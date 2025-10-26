import numpy as np
from physics.constants import MU

def orbital_period():
    '''
    Calculates the orbital period of the Earth-Moon system using Kepler's third law
    Outputs:
        T (s) - float - orbital period of the Earth-Moon system
    '''
    return 2 * np.pi

def compute_L2():
    '''
    Calculates the position and velocity of the L2 Lagrange point in the Earth-Moon system
    Outputs:
        pos_L2 (m) - (3 x 1 numpy array) - position of L2 in the barycentric frame
        v_L2 (m) - (3 x 1 numpy array) - [vx (m/s), vy (m/s), vz (m/s)] - velocity of L2 in the barycentric frame
    '''

    pos_L2 = np.array([1-MU + (MU/(3*(1-MU)))**(1/3), 0,0]) #m
    v_L2 = np.array([0, pos_L2[0], 0]) #m/s
    return pos_L2, v_L2

def pos_earth_moon(t, circular=True):
    '''
    Calculates the positions of the Earth and Moon at time t

    Inputs:
        t (dimensionless) - (float) - dimensionless time 
        circular (none) - (boolean) - indicating if the orbits are circular (default True)
    '''
    if not circular:
        raise NotImplementedError("Elliptical orbits not implemented yet.")

    T = orbital_period() 
    omega = 2 * np.pi / T 

    d_earth_barycenter = MU          
    d_moon_barycenter = 1 - MU

    x_earth_barycenter = -d_earth_barycenter * np.cos(omega * t) 
    y_earth_barycenter = -d_earth_barycenter * np.sin(omega * t) 
    x_moon_barycenter = d_moon_barycenter * np.cos(omega * t) 
    y_moon_barycenter = d_moon_barycenter * np.sin(omega * t) 

    z = np.zeros_like(x_earth_barycenter)
    r_earth = np.array([x_earth_barycenter, y_earth_barycenter, z])
    r_moon = np.array([x_moon_barycenter, y_moon_barycenter, z])

    return r_earth, r_moon
