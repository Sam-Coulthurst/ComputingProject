import numpy as np
from physics.constants import G, m_earth, m_moon
from physics.orbits import pos_earth_moon, orbital_period

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

    a_rocket_earth = -G * m_earth * r_rocket_earth / (np.linalg.norm(r_rocket_earth)**3)
    a_rocket_moon = -G * m_moon * r_rocket_moon / (np.linalg.norm(r_rocket_moon)**3)

    return a_rocket_earth + a_rocket_moon

def energy(r,m,v,r1,r2,m1 = m_earth,m2 = m_moon):
    '''
    Calculates the total energy of a mass, m, at position r, from two masses, m1 & m2 at positions r1 & r2
    respectively.

    Inputs:
        r - position vector of mass you are measuring the energy of. Relative to CoM
        m - mass of the object
        v - velocity vector of mass you are measuring the energy of. Relative to CoM
        r1 - position of larger mass 1  
        r2 - position of larger mass 2
        m1 - mass of object 1
        m2 - mass of object 2
    
    '''
    d1 = np.linalg.norm(r - r1, axis=0)
    d2 = np.linalg.norm(r - r2, axis=0)
    return 0.5 * m * np.linalg.norm(v,axis=0)**2 - G * m * (m1 / d1 + m2 / d2)

def jacobi_constant(r, v, t):
    '''
    Calculates the Jacobi constant for a given position and velocity in the Earth-Moon system

    Inputs:
        r (m) - (3 x 1 numpy array) - position vector in the barycentric frame
        v (m/s) - (3 x 1 numpy array) - velocity vector in the barycentric frame
        t (s) - (float) - time in seconds

    Outputs:
        C (J/kg) - float - Jacobi constant per unit mass
    '''
    pos_earth, pos_moon = pos_earth_moon(t)
    d_rocket_earth = np.linalg.norm(r - pos_earth, axis=0)
    d_rocket_moon = np.linalg.norm(r - pos_moon, axis=0)

    omega = 2 * np.pi / orbital_period()

    U = -G * m_earth / d_rocket_earth - G * m_moon / d_rocket_moon - 0.5 * omega**2 * np.linalg.norm(r)**2
    K = 0.5 * np.linalg.norm(v, axis=0)**2

    C = -2 * U - K
    return C
