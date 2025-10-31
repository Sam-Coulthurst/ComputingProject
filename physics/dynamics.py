import numpy as np
from physics.constants import m_earth_SI, m_moon_SI, d_earth_moon_SI, G_SI
from physics.orbits import pos_earth_moon

def acceleration(r_rocket_barycenter, t):
    '''
    Calculates the acceleration of the rocket at position r and time t

    Inputs:
        r - (N x 3 numpy array) - position vector of the rocket in the orbital plane in metres
        t - (N x 1 numpy array) - time in seconds
    '''
    pos_earth, pos_moon = pos_earth_moon(t)
    r_rocket_barycenter = np.array(r_rocket_barycenter, dtype=float)

    r_rocket_earth = r_rocket_barycenter - pos_earth
    r_rocket_moon = r_rocket_barycenter - pos_moon

    a_rocket_earth = -G_SI * m_earth_SI * r_rocket_earth / np.linalg.norm(r_rocket_earth,axis=0)**3
    a_rocket_moon = -G_SI * m_moon_SI * r_rocket_moon / np.linalg.norm(r_rocket_moon,axis=0)**3

    return a_rocket_earth + a_rocket_moon