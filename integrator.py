import numpy as np
from physics.dynamics import acceleration
from scipy.integrate import odeint, solve_ivp

def evolve(r0,v0, time, time_step, method = 'RK4', frame = 'Inertial'):
    
    # Preallocate
    n_steps = len(time)
    r = np.zeros((3, n_steps))
    v = np.zeros((3, n_steps))
    r[:, 0] = r0
    v[:, 0] = v0

    a = time_step
    
    if method == 'Taylor':
        print("Using Taylor integrator")
        for i in range(n_steps-1):
            a_rocket = acceleration(r[:, i], time[i], frame)
            
            r[:, i+1] = r[:, i] + a * v[:, i] + 0.5 * a**2 * a_rocket
            v[:, i+1] = v[:, i] + a * a_rocket

    elif method == 'RK4':
        print("Using RK4 integrator")
        for n in range(n_steps - 1):
            # z1
            r_ddot = acceleration(r[:, n], time[n], frame)
            z1_dot = v[:, n] + 0.5 * a * r_ddot
            z1 = r[:, n] + 0.5 * a * v[:, n]

            # z2
            z1_ddot = acceleration(z1, time[n] + 0.5 * a, frame)
            z2_dot = v[:, n] + 0.5 * a * z1_ddot
            z2 = r[:, n] + 0.5 * a * z1_dot

            # z3
            z2_ddot = acceleration(z2, time[n] + 0.5 * a, frame)
            z3_dot = v[:, n] + a * z2_ddot
            z3 = r[:, n] + a * z2_dot

            z3_ddot = acceleration(z3, time[n] + a, frame)

            # Combine
            r[:, n+1] = r[:, n] + (a/6) * (v[:, n] + 2*z1_dot + 2*z2_dot + z3_dot)
            v[:, n+1] = v[:, n] + (a/6) * (r_ddot + 2*z1_ddot + 2*z2_ddot + z3_ddot)
    elif method == 'odeint':
        print("Using odeint integrator")
        y0 = np.concatenate((r[:, 0], v[:, 0]))
        sol = odeint(derivatives, y0, time)
        r = sol[:, :3].T
        v = sol[:, 3:].T

    elif method == 'RK8':
        print("Using RK8 integrator")
        y0 = np.concatenate((r[:, 0], v[:, 0]))
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


def derivatives(y, t):
    # unpack
    r = y[:3]   # position vector
    v = y[3:]   # velocity vector

    # acceleration function (define this elsewhere)
    a = acceleration(r, t, frame='Inertial')
    
    # derivatives
    dydt = np.concatenate((v, a))
    return dydt

def dydt(t, y):
    return derivatives(y, t)