import numpy as np
from physics.dynamics import acceleration

def evolve(r0,v0, time, time_step, method = 'RK4'):
    
    # Preallocate
    n_steps = len(time)
    r = np.zeros((2, n_steps))
    v = np.zeros((2, n_steps))
    r[:, 0] = r0
    v[:, 0] = v0
    
    if method == 'Taylor':
        for i in range(n_steps-1):
            a_rocket = acceleration(r[:, i], time[i])
            
            r[:, i+1] = r[:, i] + time_step * v[:, i] + 0.5 * time_step**2 * a_rocket
            v[:, i+1] = v[:, i] + time_step * a_rocket

    elif method == 'RK4':
        for n in range(n_steps-1):
            # z1
            z1 = r[:,n] + 0.5*time_step*v[:,n]

            r_ddot = acceleration(r[:, n], time[n])
            z1_dot = v[:,n] + 0.5*time_step*r_ddot

            #z2
            z2 = r[:,n] + 0.5*time_step*z1_dot

            z1_ddot = acceleration(z1, time[n]+0.5*time_step)
            z2_dot = v[:,n] + 0.5*time_step*z1_ddot

            #z3
            z3 = r[:,n] + time_step*z2_dot

            z2_ddot = acceleration(z2, time[n]+0.5*time_step)
            z3_dot = v[:,n] + time_step*z2_ddot

            z3_ddot = acceleration(z3, time[n]+time_step)

            # Combine
            r[:, n+1] = r[:, n] + (1/6)*time_step*(v[:, n] + 2*z1_dot + 2*z2_dot + z3_dot)
            v[:, n+1] = v[:, n] + (1/6)*time_step*(r_ddot + 2*z1_ddot + 2*z2_ddot + z3_ddot)


    return r, v