import numpy as np
from physics.constants import G, m_earth, m_moon, d_earth_moon

def orbital_period():
    '''
    Calculates the orbital period of the Earth-Moon system using Kepler's third law
    Outputs:
        T (s) - float - orbital period of the Earth-Moon system
    '''
    return 2 * np.pi * np.sqrt(d_earth_moon**3 / (G * (m_earth + m_moon)))

def Jacobi_constant(r, v, t):
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
    d_rocket_earth = np.linalg.norm(r - pos_earth)
    d_rocket_moon = np.linalg.norm(r - pos_moon)

    omega = 2 * np.pi / orbital_period()

    U = -G * m_earth / d_rocket_earth - G * m_moon / d_rocket_moon - 0.5 * omega**2 * np.linalg.norm(r)**2
    K = 0.5 * np.linalg.norm(v)**2

    C = -2 * U - K
    return C

def compute_L2():
    '''
    Calculates the position and velocity of the L2 Lagrange point in the Earth-Moon system
    Outputs:
        pos_L2 (m) - (3 x 1 numpy array) - position of L2 in the barycentric frame
        v_L2 (m) - (3 x 1 numpy array) - [vx (m/s), vy (m/s), vz (m/s)] - velocity of L2 in the barycentric frame
    '''
    T = orbital_period()
    initial_earth, initial_moon = pos_earth_moon(0)

    d = d_earth_moon #m
    pos_L2 = np.array([initial_moon[0] + d*(m_moon/(3*m_earth))**(1/3), 0,0]) #m
    v_L2 = np.array([0, (2*np.pi/T) * pos_L2[0], 0]) #m/s
    return pos_L2, v_L2

def pos_earth_moon(t, circular=True):
    '''
    Calculates the positions of the Earth and Moon at time t

    Inputs:
        t (s) - (float) - time in seconds
        circular (none) - (boolean) - indicating if the orbits are circular (default True)
    '''
    if not circular:
        raise NotImplementedError("Elliptical orbits not implemented yet.")

    T = orbital_period() #s
    omega = 2 * np.pi / T # rad/s

    d_earth_barycenter = d_earth_moon * m_moon / (m_earth + m_moon) #m
    d_moon_barycenter = d_earth_moon * m_earth / (m_earth + m_moon) #m

    x_earth_barycenter = d_earth_barycenter * np.cos(omega * t + np.pi) #m
    y_earth_barycenter = d_earth_barycenter * np.sin(omega * t + np.pi) #m
    x_moon_barycenter = d_moon_barycenter * np.cos(omega * t) #m
    y_moon_barycenter = d_moon_barycenter * np.sin(omega * t) #m
    z = np.zeros(np.shape(x_earth_barycenter))

    return np.array([x_earth_barycenter, y_earth_barycenter,z]), np.array([x_moon_barycenter, y_moon_barycenter,z])


