import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_trajectories(r_earth_barycenter, r_moon_barycenter, r_rocket_barycenter, optimal = None):
    """
    Interactive 2D/3D plot of Earth, Moon, and Rocket trajectories using Plotly
    """
    is3d = r_earth_barycenter.shape[0] == 3

    if is3d:
        fig = go.Figure()

        # Earth
        fig.add_trace(go.Scatter3d(
            x=r_earth_barycenter[0], y=r_earth_barycenter[1], z=r_earth_barycenter[2],
            mode='lines',
            name='Earth Orbit',
            line=dict(width=4, color='green')
        ))

        # Moon
        fig.add_trace(go.Scatter3d(
            x=r_moon_barycenter[0], y=r_moon_barycenter[1], z=r_moon_barycenter[2],
            mode='lines',
            name='Moon Orbit',
            line=dict(width=2, color='blue')
        ))

        # Rocket
        fig.add_trace(go.Scatter3d(
            x=r_rocket_barycenter[0], y=r_rocket_barycenter[1], z=r_rocket_barycenter[2],
            mode='lines',
            name='Rocket Trajectory',
            line=dict(width=3, color='red')
        ))
        if optimal is not None:
            # Optimal L2 Orbit
            fig.add_trace(go.Scatter3d(
                x=optimal[0], y=optimal[1], z=optimal[2],
                mode='lines',
                name='Optimal L2 Orbit',
                line=dict(width=2, color='white', dash='dash')
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X Position (m)',
                yaxis_title='Y Position (m)',
                zaxis_title='Z Position (m)',
                xaxis=dict(range=[-2, 2], showbackground=True, backgroundcolor='black', gridcolor='gray', color='white'),
                yaxis=dict(range=[-2, 2], showbackground=True, backgroundcolor='black', gridcolor='gray', color='white'),
                zaxis=dict(range=[-2, 2], showbackground=True, backgroundcolor='black', gridcolor='gray', color='white'),
                bgcolor='black'
            ),
            title='Earth, Moon, and Rocket Trajectories',
            legend=dict(x=0.8, y=0.9),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
         )

    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=r_earth_barycenter[0], y=r_earth_barycenter[1],
            mode='lines', name='Earth Orbit', line=dict(width=4, color='green')
        ))

        fig.add_trace(go.Scatter(
            x=r_moon_barycenter[0], y=r_moon_barycenter[1],
            mode='lines', name='Moon Orbit', line=dict(width=2, color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=r_rocket_barycenter[0], y=r_rocket_barycenter[1],
            mode='lines', name='Rocket Trajectory', line=dict(width=3, color='red')
        ))

        fig.update_layout(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            title='Earth and Moon Orbits relative to Barycenter (Interactive)',
            legend=dict(x=0.8, y=0.9),
            xaxis=dict(scaleanchor='y', scaleratio=1, range=[-2, 2], color='white', gridcolor='gray'),
            yaxis=dict(range=[-2, 2], color='white', gridcolor='gray'),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
         )

    fig.show()

def animate_trajectories(r_earth_barycenter, r_moon_barycenter, r_rocket_barycenter,
                         step=100, metric=None, metric_label='Metric',
                         frame_duration=200, transition_duration=400):
    """
    Animate trajectories with a side 2D plot of `metric` vs `time`.
    - metric: array length N (or None -> rocket-to-moon distance used)
    - time: array length N (or None -> frame indices used)
    """
    def ensure_3xn(a):
        a = np.asarray(a)
        if a.ndim != 2 or a.shape[0] not in (2,3):
            raise ValueError("trajectory must be shape (2,N) or (3,N)")
        if a.shape[0] == 2:
            a = np.vstack([a, np.zeros((1, a.shape[1]), dtype=a.dtype)])
        return a

    r_earth = ensure_3xn(r_earth_barycenter)
    r_moon  = ensure_3xn(r_moon_barycenter)
    r_rocket= ensure_3xn(r_rocket_barycenter)

    n_frames = r_earth.shape[1]
    indices = np.arange(0, n_frames, max(1, step))
    
    t_axis = np.arange(n_frames)
    # prepare metric and time axes
    if metric is None:
        metric = np.linalg.norm(r_rocket - r_moon, axis=0)
    else:
        metric = np.asarray(metric)
        if metric.size != n_frames:
            raise ValueError("metric must have same length as trajectories")
        # initial side plot
        fig.add_trace(go.Scatter(x=[t_axis[0]], y=[metric[0]], mode='lines', name=metric_label,
                                line=dict(color='cyan')), row=1, col=2)
    

    # combined figure: 3D + side 2D
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'scene'}, {'type':'xy'}]], column_widths=[0.72, 0.28])

    # initial 3D traces
    fig.add_trace(go.Scatter3d(x=[r_earth[0,0]], y=[r_earth[1,0]], z=[r_earth[2,0]],
                               mode='markers', name='Earth', marker=dict(size=4, color='green')), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[r_moon[0,0]],  y=[r_moon[1,0]],  z=[r_moon[2,0]],
                               mode='markers', name='Moon',  marker=dict(size=3, color='blue')),  row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[r_rocket[0,0]],y=[r_rocket[1,0]],z=[r_rocket[2,0]],
                               mode='markers', name='Rocket',marker=dict(size=3, color='red')),   row=1, col=1)

    

    # layout (dark)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.2,1.2], showbackground=True, backgroundcolor='black', gridcolor='gray', color='white'),
            yaxis=dict(range=[-1.2,1.2], showbackground=True, backgroundcolor='black', gridcolor='gray', color='white'),
            zaxis=dict(range=[-1.2,1.2], showbackground=True, backgroundcolor='black', gridcolor='gray', color='white'),
            bgcolor='black'
        ),
        xaxis2=dict(title='Time'),
        yaxis2=dict(title=metric_label, gridcolor='gray', color='cyan'),
        title='Trajectories with side metric',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        updatemenus=[dict(type='buttons',
                          buttons=[dict(label='Play', method='animate',
                                        args=[None, {'frame': {'duration': frame_duration, 'redraw': True},
                                                     'transition': {'duration': transition_duration},
                                                     'fromcurrent': True}]),
                                   dict(label='Pause', method='animate',
                                        args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                                       'mode': 'immediate'}])])]
    )

    # build frames (order must match traces added above)
    frames = []
    for i in indices:
        idx = i + 1
        frames.append(go.Frame(data=[
            go.Scatter3d(x=r_earth[0,:idx], y=r_earth[1,:idx], z=r_earth[2,:idx]),
            go.Scatter3d(x=r_moon[0,:idx],  y=r_moon[1,:idx],  z=r_moon[2,:idx]),
            go.Scatter3d(x=r_rocket[0,:idx],y=r_rocket[1,:idx],z=r_rocket[2,:idx]),
            go.Scatter(x=t_axis[:idx], y=metric[:idx])
        ]))

    fig.frames = frames
    fig.show()

