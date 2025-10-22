import numpy as np
from physics.dynamics import acceleration
from scipy.integrate import odeint


def evolve(r0,v0, time, time_step, method = 'RK4'):
    
    # Preallocate
    n_steps = len(time)
    r = np.zeros((2, n_steps))
    v = np.zeros((2, n_steps))
    r[:, 0] = r0
    v[:, 0] = v0

    a = time_step
    
    if method == 'Taylor':
        print("Using Taylor integrator")
        for i in range(n_steps-1):
            a_rocket = acceleration(r[:, i], time[i])
            
            r[:, i+1] = r[:, i] + a * v[:, i] + 0.5 * a**2 * a_rocket
            v[:, i+1] = v[:, i] + a * a_rocket

    elif method == 'RK4':
        print("Using RK4 integrator")
        for n in range(n_steps - 1):
            # z1
            r_ddot = acceleration(r[:, n], time[n])
            z1_dot = v[:, n] + 0.5 * a * r_ddot
            z1 = r[:, n] + 0.5 * a * v[:, n]

            # z2
            z1_ddot = acceleration(z1, time[n] + 0.5 * a)
            z2_dot = v[:, n] + 0.5 * a * z1_ddot
            z2 = r[:, n] + 0.5 * a * z1_dot

            # z3
            z2_ddot = acceleration(z2, time[n] + 0.5 * a)
            z3_dot = v[:, n] + a * z2_ddot
            z3 = r[:, n] + a * z2_dot

            z3_ddot = acceleration(z3, time[n] + a)

            # Combine
            r[:, n+1] = r[:, n] + (a/6) * (v[:, n] + 2*z1_dot + 2*z2_dot + z3_dot)
            v[:, n+1] = v[:, n] + (a/6) * (r_ddot + 2*z1_ddot + 2*z2_ddot + z3_ddot)
    elif method == 'odeint':
        print("Using odeint integrator")
        y0 = np.concatenate((r[:, 0], v[:, 0]))
        sol = odeint(derivatives, y0, time)
        r = sol[:, :2].T
        v = sol[:, 2:].T
    return r, v


def derivatives(y, t):
    # unpack
    r = y[:2]   # position vector
    v = y[2:]   # velocity vector
    
    # acceleration function (define this elsewhere)
    a = acceleration(r, t)
    
    # derivatives
    dydt = np.concatenate((v, a))
    return dydt