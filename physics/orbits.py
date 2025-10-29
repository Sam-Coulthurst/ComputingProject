import numpy as np
from physics.constants import MU, L_UNIT

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
    #delta_r = (MU/(3*(1-MU)))**(1/3)
    delta_r = 64.5201e6 / L_UNIT  # dimensionless
    pos_L2 = np.array([1-MU + delta_r, 0,0]) #m
    v_L2 = np.array([0, pos_L2[0], 0]) #m/s
    return pos_L2, v_L2

def optimal_L2_orbit(pos_L2, v0_L2, t):
    '''
    Calculates an optimal orbit around L2 for a spacecraft to maintain a stable position with minimal fuel consumption.
    Inputs:
        pos_L2 (m) - (3 x 1 numpy array) - position of L2 in the barycentric frame
        v0_L2 (m/s) - (3 x 1 numpy array) - initial velocity of L2 in the barycentric frame
        '''
    r = np.linalg.norm(pos_L2)
    omega = np.linalg.norm(v0_L2) / r

    x_L2 = r * np.cos(omega*t)
    y_L2 = r * np.sin(omega*t)
    z_L2 = np.zeros_like(x_L2)

    r_L2 = np.array([x_L2, y_L2, z_L2])
    return r_L2

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
