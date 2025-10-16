import matplotlib.pyplot as plt

def plot_trajectories(d_earth_barycenter, d_moon_barycenter, d_rocket_barycenter):
    '''
    Plots the trajectories of the Earth, Moon, and Rocket relative to the barycenter
    Inputs:
        d_earth_barycenter - (2 x len(time) numpy array) - positions of the Earth relative to the barycenter
        d_moon_barycenter - (2 x len(time) numpy array) - positions of the Moon relative to the barycenter
        d_rocket_barycenter - (2 x len(time) numpy array) - positions of the Rocket relative to the barycenter

    Outputs:
        A plot showing the trajectories of the Earth, Moon, and Rocket with the barycenter at the origin
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(d_earth_barycenter[0], d_earth_barycenter[1], label='Earth Orbit')
    ax.plot(d_moon_barycenter[0], d_moon_barycenter[1], label='Moon Orbit')
    ax.plot(d_rocket_barycenter[0], d_rocket_barycenter[1], label='Rocket Trajectory')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Earth and Moon Orbits relative to Barycenter')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(-5e8, 5e8)
    ax.set_ylim(-5e8, 5e8)
    ax.set_aspect('equal', 'box')
    plt.show()
