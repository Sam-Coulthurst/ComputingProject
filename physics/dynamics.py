import numpy as np
from physics.constants import G, m_earth, m_moon
from physics.orbits import pos_earth_moon

def acceleration(r_rocket_barycenter, t):
    '''
    Calculates the acceleration of the rocket at position r and time t

    Inputs:
        r (m) - (2x1 numpy array) - position vector of the rocket in the orbital plane
        t (s) - (float) - time in seconds
    '''
    pos_earth, pos_moon = pos_earth_moon(t)
    r_rocket_earth = r_rocket_barycenter - pos_earth
    r_rocket_moon = r_rocket_barycenter - pos_moon

    a_rocket_earth = -G * m_earth * r_rocket_earth / np.linalg.norm(r_rocket_earth)**3
    a_rocket_moon = -G * m_moon * r_rocket_moon / np.linalg.norm(r_rocket_moon)**3

    return a_rocket_earth + a_rocket_moon
