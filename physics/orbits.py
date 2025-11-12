import numpy as np
from physics.constants import G_SI, m_earth_SI, m_moon_SI, d_earth_moon_SI

def orbital_period():
    '''
    Calculates the orbital period of the Earth-Moon system using Kepler's third law
    Outputs:
        T (s) - float - orbital period of the Earth-Moon system
    '''
    return 2 * np.pi * np.sqrt(d_earth_moon_SI**3/(G_SI * (m_earth_SI + m_moon_SI)))

def compute_L2(alpha=1):
    '''
    Calculates the position and velocity of the L2 Lagrange point in the Earth-Moon system

    Inputs:
        alpha (float) - scaling factor for delta_r calculation

    Outputs:
        pos_L2 (m) - (3 x 1 numpy array) - position of L2 in the barycentric frame
        v_L2 (m) - (3 x 1 numpy array) - [vx (m/s), vy (m/s), vz (m/s)] - velocity of L2 in the barycentric frame
    '''
    delta_r = alpha * d_earth_moon_SI * (m_moon_SI/(3*m_earth_SI))**(1/3)
    #print(f'Optimal delta_r: {delta_r*1e-3} km')
    pos_moon = (m_earth_SI/(m_earth_SI + m_moon_SI)) * d_earth_moon_SI
    
    pos_L2 = np.array([pos_moon + delta_r, 0,0]) 

    v_L2 = np.array([0, 2*np.pi*pos_L2[0]/orbital_period(), 0]) 
    
    return pos_L2, v_L2

def optimal_L2_orbit(pos_L2,  t):
    '''
    Calculates an optimal orbit around L2 for a spacecraft to maintain a stable position with minimal fuel consumption.
    Inputs:
        pos_L2 (m) - (N x 3 numpy array) - position of L2 in the barycentric frame
        v0_L2 (m/s) - (N x 3 numpy array) - initial velocity of L2 in the barycentric frame
        '''
    r = np.linalg.norm(pos_L2)
    omega = 2 * np.pi / orbital_period()

    x_L2 = r * np.cos(omega*t)
    y_L2 = r * np.sin(omega*t)
    z_L2 = np.zeros_like(x_L2)

    r_L2 = np.stack((x_L2, y_L2, z_L2), axis=0).T
    return r_L2

def pos_earth_moon(t, circular=True):
    '''
    Calculates the positions of the Earth and Moon at time t

    Inputs:
        t (s) - (float) - time in seconds
        circular (none) - (boolean) - indicating if the orbits are circular (default True)

    Outputs:
        pos_earth (m) - (N x 3 numpy array) - position of the Earth in the barycentric frame
        pos_moon (m) - (N x 3 numpy array) - position of the Moon in the barycentric frame
    '''
    if not circular:
        raise NotImplementedError("Elliptical orbits not implemented yet.")

    T = orbital_period() 
    omega = 2 * np.pi / T 

    d_earth_barycenter = d_earth_moon_SI * m_moon_SI / (m_earth_SI + m_moon_SI)          
    d_moon_barycenter = d_earth_moon_SI * m_earth_SI / (m_earth_SI + m_moon_SI) 

    x_earth_barycenter = -d_earth_barycenter * np.cos(omega * t) 
    y_earth_barycenter = -d_earth_barycenter * np.sin(omega * t) 
    x_moon_barycenter = d_moon_barycenter * np.cos(omega * t) 
    y_moon_barycenter = d_moon_barycenter * np.sin(omega * t) 

    z = np.zeros_like(x_earth_barycenter)
    r_earth = np.stack([x_earth_barycenter, y_earth_barycenter, z], axis=-1)
    r_moon = np.stack([x_moon_barycenter, y_moon_barycenter, z], axis=-1)

    return r_earth, r_moon
