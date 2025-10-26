import plotly.graph_objects as go
import numpy as np

def animate_trajectories(d_earth_barycenter, d_moon_barycenter, d_rocket_barycenter):
    is3d = d_earth_barycenter.shape[0] == 3
    n_frames = d_earth_barycenter.shape[1]

    if is3d:
        fig = go.Figure(
            data=[
                go.Scatter3d(x=[d_earth_barycenter[0,0]], y=[d_earth_barycenter[1,0]], z=[d_earth_barycenter[2,0]],
                             mode='markers', name='Earth', marker=dict(size=6, color='blue')),
                go.Scatter3d(x=[d_moon_barycenter[0,0]], y=[d_moon_barycenter[1,0]], z=[d_moon_barycenter[2,0]],
                             mode='markers', name='Moon', marker=dict(size=4, color='gray')),
                go.Scatter3d(x=[d_rocket_barycenter[0,0]], y=[d_rocket_barycenter[1,0]], z=[d_rocket_barycenter[2,0]],
                             mode='markers', name='Rocket', marker=dict(size=5, color='red')),
            ],
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(range=[-1.2, 1.2]),
                    yaxis=dict(range=[-1.2, 1.2]),
                    zaxis=dict(range=[-1.2, 1.2]),
                ),
                title='Earth, Moon, and Rocket Trajectories (Animated)',
                updatemenus=[dict(
                    type='buttons',
                    buttons=[dict(label='Play',
                                  method='animate',
                                  args=[None, {'frame': {'duration':50, 'redraw':True},
                                               'fromcurrent':True}]),
                             dict(label='Pause',
                                  method='animate',
                                  args=[[None], {'frame': {'duration':0, 'redraw':False},
                                                 'mode':'immediate'}])]
                )]
            )
        )

        frames = []
        for i in range(n_frames):
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=d_earth_barycenter[0,:i+1], y=d_earth_barycenter[1,:i+1], z=d_earth_barycenter[2,:i+1]),
                    go.Scatter3d(x=d_moon_barycenter[0,:i+1], y=d_moon_barycenter[1,:i+1], z=d_moon_barycenter[2,:i+1]),
                    go.Scatter3d(x=d_rocket_barycenter[0,:i+1], y=d_rocket_barycenter[1,:i+1], z=d_rocket_barycenter[2,:i+1]),
                ]
            ))

        fig.frames = frames
        fig.show()
