import numpy as np
from physics.constants import MU
from physics.orbits import pos_earth_moon, orbital_period

def acceleration(r_rocket_barycenter, t, v = None, frame='Inertial'):
    '''
    Calculates the acceleration of the rocket at position r and time t

    Inputs:
        r - (2x1 numpy array) - position vector of the rocket in the orbital plane
        t - (float) - time in seconds
    '''
    if frame == 'Inertial':
        pos_earth, pos_moon = pos_earth_moon(t)
        r_rocket_barycenter = np.array(r_rocket_barycenter, dtype=float)

        r_rocket_earth = r_rocket_barycenter - pos_earth
        r_rocket_moon = r_rocket_barycenter - pos_moon

        a_rocket_earth = -(1.0 - MU) * r_rocket_earth / np.linalg.norm(r_rocket_earth)**3
        a_rocket_moon = -MU * r_rocket_moon / np.linalg.norm(r_rocket_moon)**3
        return a_rocket_earth + a_rocket_moon

    elif frame == 'Rotating':
        Omega = np.array([0, 0, 1.0])  # normalized canonical angular rate
        r1 = np.array([-MU, 0, 0])
        r2 = np.array([1 - MU, 0, 0])
        a_grav = -(1 - MU) * (r_rocket_barycenter - r1) / np.linalg.norm(r_rocket_barycenter - r1)**3 \
                 - MU * (r_rocket_barycenter - r2) / np.linalg.norm(r_rocket_barycenter - r2)**3

        # Coriolis and centrifugal
        coriolis = -2 * np.cross(Omega, v)
        centrifugal = -np.cross(Omega, np.cross(Omega, r_rocket_barycenter))
        return a_grav + coriolis + centrifugal
        
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


def jacobi_constant(r, v):
    """
    Compute the Jacobi constant in the circular restricted three-body problem.
    
    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector (x, y, z) in the rotating frame.
    v : array_like, shape (3,)
        Velocity vector (vx, vy, vz) in the rotating frame.
    mu : float
        Mass parameter mu = m2 / (m1 + m2).
    
    Returns
    -------
    C : float
        Jacobi constant.
    """
    x, y, z = r
    vx, vy, vz = v
    
    r1 = np.sqrt((x + MU)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + MU)**2 + y**2 + z**2)

    U = 0.5*(x**2 + y**2) + (1 - MU)/r1 + MU/r2
    C = 2*U - (vx**2 + vy**2 + vz**2)
    return C


def rotate_to_inertial(r_rot, v_rot, t):
    """
    Rotate from rotating frame to inertial frame for 3xN arrays.

    Parameters
    ----------
    r_rot : np.ndarray
        (3, N) array of position vectors in rotating frame
    v_rot : np.ndarray
        (3, N) array of velocity vectors in rotating frame
    t : float or np.ndarray
        scalar or (N,) array of rotation angles (radians, typically dimensionless time)

    Returns
    -------
    r_I : np.ndarray
        (3, N) array of inertial-frame position vectors
    v_I : np.ndarray
        (3, N) array of inertial-frame velocity vectors
    """
    t = np.atleast_1d(t)
    N = r_rot.shape[1]

    # Broadcast rotation angles
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    # Preallocate
    r_I = np.zeros_like(r_rot)
    v_I = np.zeros_like(v_rot)

    # Rotate positions and velocities
    r_I[0, :] = cos_t * r_rot[0, :] - sin_t * r_rot[1, :]
    r_I[1, :] = sin_t * r_rot[0, :] + cos_t * r_rot[1, :]
    r_I[2, :] = r_rot[2, :]

    v_I[0, :] = cos_t * v_rot[0, :] - sin_t * v_rot[1, :]
    v_I[1, :] = sin_t * v_rot[0, :] + cos_t * v_rot[1, :]
    v_I[2, :] = v_rot[2, :]

    # Add rotational contribution: ω × r_I, with ω = [0, 0, 1]
    omega_cross_rI = np.zeros_like(r_I)
    omega_cross_rI[0, :] = -r_I[1, :]
    omega_cross_rI[1, :] =  r_I[0, :]
    # z component = 0

    v_I += omega_cross_rI

    return r_I, v_I