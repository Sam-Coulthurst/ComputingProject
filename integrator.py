import numpy as np
from physics.dynamics import acceleration
from scipy.integrate import odeint, solve_ivp

def evolve(r0,v0, time, EoM, method = 'RK4'):
    '''
    Evolves the equations of motion for a rocket under given acceleration function EoM
    Inputs:
        r0 (m) - (3 x 1 numpy array) - initial position of the rocket in the barycentric frame
        v0 (m/s) - (3 x 1 numpy array) - initial velocity of the rocket in the barycentric frame
        time (s) - (1 x N numpy array) - time array for evolution
        EoM - function - function that computes acceleration given position and time
        method - string - integration method to use ('Taylor', 'RK4', 'odeint', 'RK8')
    Outputs:
        r (m) - (N x 3 numpy array) - position of the rocket at each time step
        v (m/s) - (N x 3 numpy array) - velocity of the rocket at each time step
    '''
    # Preallocate
    n_steps = len(time)
    r = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    r[0] = r0
    v[0] = v0

    a = time[1] - time[0]  # time step size
    
    if method == 'Taylor':
        for i in range(n_steps-1):
            a_rocket = EoM(r[i], time[i])

            r[i+1] = r[i] + a * v[i] + 0.5 * a**2 * a_rocket
            v[i+1] = v[i] + a * a_rocket

    elif method == 'RK4':
        for n in range(n_steps - 1):
            # z1
            r_ddot = EoM(r[n], time[n])
            z1_dot = v[n] + 0.5 * a * r_ddot
            z1 = r[n] + 0.5 * a * v[n]

            # z2
            z1_ddot = EoM(z1, time[n] + 0.5 * a)
            z2_dot = v[n] + 0.5 * a * z1_ddot
            z2 = r[n] + 0.5 * a * z1_dot

            # z3
            z2_ddot = EoM(z2, time[n] + 0.5 * a)
            z3_dot = v[n] + a * z2_ddot
            z3 = r[n] + a * z2_dot

            z3_ddot = EoM(z3, time[n] + a)

            # Combine
            r[n+1] = r[n] + (a/6) * (v[n] + 2*z1_dot + 2*z2_dot + z3_dot)
            v[n+1] = v[n] + (a/6) * (r_ddot + 2*z1_ddot + 2*z2_ddot + z3_ddot)

    elif method == 'odeint':
        y0 = np.concatenate((r[0], v[0]))
        sol = odeint(derivatives, y0, time)
        r = sol[:, :3]
        v = sol[:, 3:]

    elif method == 'RK8':
        y0 = np.concatenate((r[0], v[0]))
        sol = solve_ivp(
            dydt,
            (time[0], time[-1]),
            y0,
            t_eval=time,
            method='DOP853',
            rtol=1e-9,
            atol=1e-12
        )
        r = sol.y[:3, :]
        v = sol.y[3:, :]

    return r, v


def derivatives(y, t, EoM=acceleration):
    # unpack
    r = y[:3]   # position vector
    v = y[3:]   # velocity vector

    # acceleration function (define this elsewhere)
    a = EoM(r, t)
    
    # derivatives
    dydt = np.concatenate((v, a))
    return dydt

def dydt(t, y):
    return derivatives(y, t)