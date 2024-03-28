import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

from app.Color import Color

def add_curve(y, color=None, x = None, name = "", fig = None, row = None, col = None):
    color = color if color is not None else f'rgb{Color.next()}'
    Y = np.array(y)
    X = np.array(x) if x is not None else np.array(list(range(Y.size)))
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=Y, name=name, line_color=color), row=row, col=col)
    return fig


def add_aplat(under, above, color=None, x = None, name = "", fig = None, row = None, col = None):
    color = color if color is not None else f'rgb{Color.next()}'

    UNDER = np.array(under)
    ABOVE = np.array(above)
    X = np.array(x) if x is not None else np.array(list(range(ABOVE.size)))
    if fig is None:
        fig = go.Figure()

    fig.add_trace(go.Scatter(
        name=f'{name} Upper Bound',
        x=X,
        y=ABOVE,
        mode='lines',
        line_width=0,
        showlegend=False
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        name=f'{name} Lower Bound',
        x=X,
        y=UNDER,
        line=dict(width=0),
        mode='lines',
        fill='tonexty',
        fillcolor=color,
        showlegend=False
    ), row=row, col=col)

    return fig


def magnitude_angle(
        frame_times,
        magnitude_means,
        magnitude_stds,
        angle_means,
        angle_stds,
        colors = None,
        fig=make_subplots(rows=2, cols=1),
        doShow=False,
        rows=[1,2],
        cols=[1,1]
):
    
    colors = colors if colors is not None else (Color.next(), Color.next())
    add_curve(magnitude_means, f'rgb{colors[0]}', x=frame_times, name='amplitude=f(t) - Amplitude du flux', fig=fig, row=rows[0], col=cols[0])
    add_curve(angle_means, f'rgb{colors[1]}', x=frame_times, name='angle=f(t) - Angle du flux', fig=fig, row=rows[1], col=cols[1])

    add_aplat(magnitude_means-magnitude_stds, magnitude_means+magnitude_stds, f'rgba({colors[0][0]},{colors[0][1]},{colors[0][2]},0.3)', x=frame_times, name='Magnitude', fig=fig, row=rows[0], col=cols[0])
    add_aplat(angle_means-angle_stds, angle_means+angle_stds, f'rgba({colors[1][0]},{colors[1][1]},{colors[1][2]},0.3)', x=frame_times, name='Angle', fig=fig, row=rows[1], col=cols[1])

    if doShow: fig.show()
    return fig