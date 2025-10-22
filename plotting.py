import matplotlib.pyplot as plt

import plotly.graph_objects as go

def plot_trajectories(d_earth_barycenter, d_moon_barycenter, d_rocket_barycenter):
    """
    Interactive 2D/3D plot of Earth, Moon, and Rocket trajectories using Plotly
    """
    is3d = d_earth_barycenter.shape[0] == 3

    if is3d:
        fig = go.Figure()

        # Earth
        fig.add_trace(go.Scatter3d(
            x=d_earth_barycenter[0], y=d_earth_barycenter[1], z=d_earth_barycenter[2],
            mode='lines',
            name='Earth Orbit',
            line=dict(width=4, color='blue')
        ))

        # Moon
        fig.add_trace(go.Scatter3d(
            x=d_moon_barycenter[0], y=d_moon_barycenter[1], z=d_moon_barycenter[2],
            mode='lines',
            name='Moon Orbit',
            line=dict(width=2, color='gray')
        ))

        # Rocket
        fig.add_trace(go.Scatter3d(
            x=d_rocket_barycenter[0], y=d_rocket_barycenter[1], z=d_rocket_barycenter[2],
            mode='lines',
            name='Rocket Trajectory',
            line=dict(width=3, color='red')
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X Position (m)',
                yaxis_title='Y Position (m)',
                zaxis_title='Z Position (m)',
                xaxis=dict(range=[-5e8, 5e8]),
                yaxis=dict(range=[-5e8, 5e8]),
                zaxis=dict(range=[-5e8, 5e8]),
            ),
            title='Earth, Moon, and Rocket Trajectories (Interactive)',
            legend=dict(x=0.8, y=0.9)
        )

    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=d_earth_barycenter[0], y=d_earth_barycenter[1],
            mode='lines', name='Earth Orbit', line=dict(width=4, color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=d_moon_barycenter[0], y=d_moon_barycenter[1],
            mode='lines', name='Moon Orbit', line=dict(width=2, color='gray')
        ))

        fig.add_trace(go.Scatter(
            x=d_rocket_barycenter[0], y=d_rocket_barycenter[1],
            mode='lines', name='Rocket Trajectory', line=dict(width=3, color='red')
        ))

        fig.update_layout(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            title='Earth and Moon Orbits relative to Barycenter (Interactive)',
            legend=dict(x=0.8, y=0.9),
            xaxis=dict(scaleanchor='y', scaleratio=1, range=[-5e8, 5e8]),
            yaxis=dict(range=[-5e8, 5e8])
        )

    fig.show()



