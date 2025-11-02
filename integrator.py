import numpy as np
from scipy.integrate import odeint, solve_ivp

def evolve(r0, v0, time, EoM, method='RK4'):
    '''
    Evolves the equations of motion for a rocket under given acceleration function EoM.
    EoM(r, v, t) -> acceleration vector (np.array).
    '''
    n_steps = len(time)
    dim = len(r0)
    r = np.zeros((n_steps, dim))
    v = np.zeros((n_steps, dim))
    r[0] = r0
    v[0] = v0

    dt = time[1] - time[0]

    if method == 'Taylor':
        for i in range(n_steps - 1):
            a_rocket = EoM(r[i], time[i])
            r[i + 1] = r[i] + dt * v[i] + 0.5 * dt**2 * a_rocket
            v[i + 1] = v[i] + dt * a_rocket

    elif method == 'RK4':
        for n in range(n_steps - 1):
            y_n = np.concatenate((r[n], v[n]))
            k1 = derivatives(time[n], y_n, EoM)
            k2 = derivatives(time[n] + 0.5 * dt, y_n + 0.5 * dt * k1, EoM)
            k3 = derivatives(time[n] + 0.5 * dt, y_n + 0.5 * dt * k2, EoM)
            k4 = derivatives(time[n] + dt, y_n + dt * k3, EoM)
            y_next = y_n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            r[n + 1] = y_next[:dim]
            v[n + 1] = y_next[dim:]

    elif method == 'odeint':
        y0 = np.concatenate((r0, v0))
        sol = odeint(lambda y, t: derivatives(t, y, EoM), y0, time)
        r = sol[:, :dim]
        v = sol[:, dim:]

    elif method == 'RK8':
        y0 = np.concatenate((r0, v0))
        sol = solve_ivp(
            lambda t, y: derivatives(t, y, EoM),
            (time[0], time[-1]),
            y0,
            t_eval=time,
            method='DOP853',
            rtol=1e-9,
            atol=1e-12
        )
        r = sol.y[:dim, :].T
        v = sol.y[dim:, :].T

    else:
        raise ValueError(f"Unknown integration method '{method}'")

    return r, v


def derivatives(t, y, EoM):
    """Compute dy/dt = [v, a(r, v, t)] for the given state y."""
    dim = len(y) // 2
    r = y[:dim]
    v = y[dim:]
    a = EoM(r, t)
    return np.concatenate((v, a))
