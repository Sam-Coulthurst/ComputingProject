import numpy as np
from physics.constants import MU
from physics.orbits import pos_earth_moon, orbital_period

def acceleration(r_rocket_barycenter, t):
    '''
    Calculates the acceleration of the rocket at position r and time t

    Inputs:
        r (m) - (2x1 numpy array) - position vector of the rocket in the orbital plane
        t (s) - (float) - time in seconds
    '''
    pos_earth, pos_moon = pos_earth_moon(t)
    r_rocket_barycenter = np.array(r_rocket_barycenter, dtype=float)

    r_rocket_earth = r_rocket_barycenter - pos_earth
    r_rocket_moon = r_rocket_barycenter - pos_moon

    a_rocket_earth = -(1.0 - MU) * r_rocket_earth / np.linalg.norm(r_rocket_earth)**3
    a_rocket_moon = -MU * r_rocket_moon / np.linalg.norm(r_rocket_moon)**3

    return a_rocket_earth + a_rocket_moon

def energy(r,v,t):
    '''
    Calculates the total energy of a mass, m, at position r, from two masses, m1 & m2 at positions r1 & r2
    respectively.

    Inputs:
        r - position vector of mass you are measuring the energy of. Relative to CoM
        v - velocity vector of mass you are measuring the energy of. Relative to CoM
        t - time
    
    '''
    pos_earth, pos_moon = pos_earth_moon(t)
    d1 = np.linalg.norm(r - pos_earth, axis=0)
    d2 = np.linalg.norm(r - pos_moon, axis=0)

    U = -(1 - MU) / d1 - MU / d2

    K = 0.5 * np.linalg.norm(v)**2

    return K + U