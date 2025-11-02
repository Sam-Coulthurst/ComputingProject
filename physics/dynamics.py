import numpy as np
from physics.constants import m_earth_SI, m_moon_SI, d_earth_moon_SI, G_SI
from physics.orbits import pos_earth_moon

def acceleration(r_rocket_barycenter, t):
    r_rocket_barycenter = np.array(r_rocket_barycenter, dtype=float)
    pos_earth, pos_moon = pos_earth_moon(t)

    # Handle both single and multiple vectors
    if r_rocket_barycenter.ndim == 1:
        # Single vector case
        r_rocket_earth = r_rocket_barycenter - pos_earth
        r_rocket_moon  = r_rocket_barycenter - pos_moon

        dist_earth = np.linalg.norm(r_rocket_earth)
        dist_moon  = np.linalg.norm(r_rocket_moon)

        a_rocket_earth = -G_SI * m_earth_SI * r_rocket_earth / dist_earth**3
        a_rocket_moon  = -G_SI * m_moon_SI  * r_rocket_moon  / dist_moon**3

    else:
        # Multiple vectors (N,3)
        r_rocket_earth = r_rocket_barycenter - pos_earth
        r_rocket_moon  = r_rocket_barycenter - pos_moon

        dist_earth = np.linalg.norm(r_rocket_earth, axis=1)
        dist_moon  = np.linalg.norm(r_rocket_moon,  axis=1)

        a_rocket_earth = -G_SI * m_earth_SI * r_rocket_earth / dist_earth[:, None]**3
        a_rocket_moon  = -G_SI * m_moon_SI  * r_rocket_moon  / dist_moon[:, None]**3

    return a_rocket_earth + a_rocket_moon


